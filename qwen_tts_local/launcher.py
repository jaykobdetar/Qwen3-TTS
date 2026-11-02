# SPDX-License-Identifier: Apache-2.0
"""Local Linux start/stop CLI with an inherited lock and pidfd-bound signals.

This module has no Torch/Gradio imports until the model child is explicitly started.
No resource watchdog, background service, shell execution or broad process killing.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
import webbrowser

from .config import Settings, default_config_path, load_settings, write_template


def require_linux():
    if (
        sys.platform != "linux"
        or not hasattr(os, "pidfd_open")
        or not hasattr(signal, "pidfd_send_signal")
    ):
        raise RuntimeError(
            "This launcher requires Linux with pidfd support; use qwen-tts-demo directly elsewhere (此启动器需要支持 pidfd 的 Linux；其他系统请直接使用 qwen-tts-demo)."
        )


def identity(pid):
    """Identify one live same-user process, including PID reuse and machine reboot."""
    proc = Path("/proc") / str(pid)
    stat = (proc / "stat").read_text()
    fields = stat.rsplit(") ", 1)[1].split()
    if fields[0] == "Z" or proc.stat().st_uid != os.getuid():
        raise ProcessLookupError(pid)
    return {
        "pid": pid,
        "start_ticks": fields[19],
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "uid": proc.stat().st_uid,
        "exe": os.readlink(proc / "exe"),
        "argv": (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0"),
    }


def verified_pidfd(record):
    """Open the kernel handle BEFORE matching identity; signal only through it."""
    expected = record["identity"]
    pid = expected["pid"]
    argv = expected["argv"]
    if (
        type(pid) is not int
        or pid <= 1
        or len(argv) < 8
        or argv[1:4] != ["-m", "qwen_tts_local.launcher", "_serve"]
    ):
        raise ValueError("Not a Qwen launcher identity (不是 Qwen 启动器身份).")
    if argv[4:6] != ["--instance", record["instance"]]:
        raise ValueError("Instance identity mismatch (实例身份不匹配).")
    fd = os.pidfd_open(pid)
    try:
        if identity(pid) != expected:
            raise ProcessLookupError(
                "Process identity changed; no signal sent (进程身份已更改，未发送信号)."
            )
    except BaseException:
        os.close(fd)
        raise
    return fd


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        json.dump(value, out, indent=2)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def read_record(root):
    return json.loads((root / "run.json").read_text(encoding="utf-8"))


def remove_record(root, instance):
    try:
        if read_record(root)["instance"] == instance:
            (root / "run.json").unlink()
    except FileNotFoundError:
        pass


def acquire_lock(root):
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "run.lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


def start(settings, open_browser=None):
    require_linux()
    settings.validate()
    root = settings.state_path
    try:
        lock = acquire_lock(root)
    except BlockingIOError:
        # Never load another model or open an unverified service when startup races.
        return (
            "Already running or starting; use status (已运行或正在启动，请使用 status)."
        )
    with lock:
        try:
            existing = read_record(root)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            try:
                fd = verified_pidfd(existing)
            except (FileNotFoundError, ProcessLookupError):
                pass  # positively stale record; lock is free
            else:
                os.close(fd)
                return "Verified instance already running (已验证的实例正在运行)."
        with socket.socket() as probe:
            probe.bind(
                ("127.0.0.1", settings.port)
            )  # refuse occupied port; never kill its owner
        instance = uuid.uuid4().hex
        snapshot = root / ("settings-" + instance + ".json")
        atomic_json(snapshot, asdict(settings))
        ready_read, ready_write = os.pipe()
        command = [
            sys.executable,
            "-m",
            "qwen_tts_local.launcher",
            "_serve",
            "--instance",
            instance,
            "--snapshot",
            str(snapshot),
            "--lock-fd",
            str(lock.fileno()),
            "--ready-fd",
            str(ready_read),
        ]
        try:
            with (root / "gui.log").open("a", encoding="utf-8") as log:
                child = subprocess.Popen(
                    command,
                    env=settings.environment(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(), ready_read),
                    start_new_session=True,
                )
            record = {
                "instance": instance,
                "identity": identity(child.pid),
                "url": settings.url,
            }
            atomic_json(root / "run.json", record)
            os.write(
                ready_write, b"1"
            )  # child imports the model only after identity is durably recorded
        finally:
            os.close(ready_read)
            os.close(
                ready_write
            )  # EOF makes an unrecorded child exit if startup failed
    # The child inherited the lock: launcher exit/crash cannot permit a duplicate model.
    deadline = time.monotonic() + settings.startup_wait_seconds
    should_open = settings.open_browser if open_browser is None else open_browser
    while time.monotonic() < deadline:
        if child.poll() is not None:
            remove_record(root, instance)
            return f"Model exited with code {child.returncode}; inspect {root / 'gui.log'} (模型已退出，请检查日志)."
        try:
            local_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with local_http.open(settings.url + "/config", timeout=0.3) as response:
                ready = response.status == 200
            fd = verified_pidfd(record)
            os.close(fd)
            if ready:
                if should_open:
                    webbrowser.open(settings.url)
                return "Ready (已就绪): " + settings.url
        except (OSError, ValueError):
            pass
        time.sleep(0.25)
    return (
        "Still starting; no process was terminated. Use status and gui.log (仍在启动，未终止进程。请检查 status 和日志): "
        + settings.url
    )


def stop(settings, force=False):
    require_linux()
    root = settings.state_path
    try:
        record = read_record(root)
        fd = verified_pidfd(record)
    except (FileNotFoundError, ProcessLookupError):
        return "No matching running instance; no signal sent (未找到匹配的运行实例，未发送信号)."
    try:
        try:
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        except ProcessLookupError:
            remove_record(root, record["instance"])
            return "Verified instance already exited (已验证的实例已经退出)."
        if not select.select([fd], [], [], settings.stop_wait_seconds)[0]:
            if not force:
                return "Still stopping; use stop --force only if needed (仍在停止，必要时使用 stop --force)."
            try:
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not select.select([fd], [], [], settings.stop_wait_seconds)[0]:
                return "Exit is still pending (退出仍未完成)."
        remove_record(root, record["instance"])
        return "Stopped verified Qwen instance (已停止经过身份验证的 Qwen 实例)."
    finally:
        os.close(fd)


def status(settings):
    require_linux()
    try:
        record = read_record(settings.state_path)
        fd = verified_pidfd(record)
    except (FileNotFoundError, ProcessLookupError):
        return "No matching running instance (未找到匹配的运行实例)."
    os.close(fd)
    return f"Running PID {record['identity']['pid']} (正在运行): {record['url']}"


def serve(args):
    require_linux()
    # Inherited descriptors are required. Direct invocation cannot bypass the lock/handshake.
    os.fstat(args.lock_fd)
    try:
        if os.read(args.ready_fd, 1) != b"1":
            return 1
    finally:
        os.close(args.ready_fd)
    settings = Settings(
        **json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    ).validate()
    record = read_record(settings.state_path)
    if (
        record["instance"] != args.instance
        or identity(os.getpid()) != record["identity"]
    ):
        raise RuntimeError("Child identity mismatch (子进程身份不匹配).")
    # Import only now, after the parent created a recoverable process record.
    from qwen_tts.cli.demo import main as demo_main

    try:
        return demo_main(settings.demo_args())
    finally:
        remove_record(settings.state_path, args.instance)
        Path(args.snapshot).unlink(missing_ok=True)
        os.close(args.lock_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Qwen local start/stop (Qwen 本地启动/停止)"
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser(
        "init",
        help="Write a config template without overwriting (创建配置模板，不覆盖现有文件)",
    )
    launch = commands.add_parser(
        "start", help="Start one local model (启动单个本地模型)"
    )
    launch.add_argument("--no-browser", action="store_true")
    terminate = commands.add_parser(
        "stop", help="Stop only the verified instance (仅停止身份匹配的实例)"
    )
    terminate.add_argument("--force", action="store_true")
    commands.add_parser("status")
    internal = commands.add_parser("_serve", help=argparse.SUPPRESS)
    internal.add_argument("--instance", required=True)
    internal.add_argument("--snapshot", required=True)
    internal.add_argument("--lock-fd", type=int, required=True)
    internal.add_argument("--ready-fd", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "_serve":
            return serve(args)
        if args.action == "init":
            write_template(args.config)
            print("Edit checkpoint and settings (请编辑模型路径和配置):", args.config)
            return 0
        settings = load_settings(args.config)
        if args.action == "start":
            result = start(settings, False if args.no_browser else None)
        elif args.action == "stop":
            result = stop(settings, args.force)
        else:
            result = status(settings)
        print(result)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print("Local launcher error (本地启动器错误): " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
