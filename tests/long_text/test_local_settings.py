# SPDX-License-Identifier: Apache-2.0
"""Configuration and process identity tests. No Qwen model or Torch imports."""

import ast
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qwen_tts_local.config import Settings, apply_runtime, load_settings, write_template
from qwen_tts_local.language import translator
from qwen_tts_local import launcher


class ConfigurationTests(unittest.TestCase):
    def test_local_defaults_and_environment(self):
        s = Settings(checkpoint="Qwen/cached-model")
        self.assertEqual(s.ui_language, "en")
        self.assertIn("127.0.0.1", s.demo_args())
        self.assertIn("--no-share", s.demo_args())
        self.assertEqual(s.environment()["HF_HUB_OFFLINE"], "1")
        self.assertEqual(s.environment()["OMP_NUM_THREADS"], "2")
        self.assertEqual(replace(s, offline=False).environment()["HF_HUB_OFFLINE"], "0")

    def test_invalid_config_refused(self):
        for change in (
            dict(checkpoint=None),
            dict(port=80),
            dict(threads=0),
            dict(interop_threads=True),
            dict(ui_language="Chinese"),
            dict(gpu_memory_fraction=0),
            dict(gpu_memory_fraction=float("nan")),
            dict(section_max_new_tokens=-1),
            dict(device="auto"),
            dict(attention="unknown"),
            dict(offline="yes"),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(Settings(checkpoint="cached/model"), **change).validate()

    def test_config_paths_and_template_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "local.json"
            write_template(p)
            with self.assertRaises(FileExistsError):
                write_template(p)
            p.write_text(
                json.dumps(
                    dict(checkpoint="./model", state_dir="run", projects_dir="projects")
                )
            )
            s = load_settings(p)
            self.assertEqual(s.checkpoint, str(Path(d) / "model"))
            self.assertEqual(s.state_path, Path(d) / "run")
            self.assertEqual(s.projects_path, Path(d) / "projects")
            p.write_text('{"unknown": 1}')
            with self.assertRaises(ValueError):
                load_settings(p)

    def test_runtime_settings_applied_only_to_selected_device(self):
        torch = Mock()
        apply_runtime(torch, "cuda:1", 3, 2, 0.5)
        torch.set_num_threads.assert_called_once_with(3)
        torch.set_num_interop_threads.assert_called_once_with(2)
        torch.device.assert_called_once_with("cuda:1")
        torch.cuda.set_per_process_memory_fraction.assert_called_once_with(
            0.5, torch.device.return_value
        )
        cpu = Mock()
        apply_runtime(cpu, "cpu", 2, 1, 0.7)
        cpu.cuda.set_per_process_memory_fraction.assert_not_called()
        with self.assertRaises(ValueError):
            apply_runtime(cpu, "cpu", threads=-2)

    def test_cli_accepts_launcher_options_without_loading_model(self):
        tree = ast.parse((ROOT / "qwen_tts/cli/demo.py").read_text())
        parser_function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "build_parser"
        )
        import argparse

        ns = {"argparse": argparse}
        exec(
            compile(
                ast.Module(body=[parser_function], type_ignores=[]),
                "demo-parser",
                "exec",
            ),
            ns,
        )
        s = Settings(
            checkpoint="cached/model",
            ui_language="en",
            threads=4,
            section_max_new_tokens=128,
        )
        args = ns["build_parser"]().parse_args(s.demo_args())
        self.assertFalse(args.share)
        self.assertEqual(
            (args.ip, args.ui_language, args.threads, args.long_text_max_new_tokens),
            ("127.0.0.1", "en", 4, 128),
        )
        defaults = ns["build_parser"]().parse_args(["cached/model"])
        self.assertEqual(
            (defaults.ip, defaults.ui_language), ("127.0.0.1", "bilingual")
        )

    def test_translation_preserves_user_text_and_language_support(self):
        en, bi = translator("en"), translator("bilingual")
        self.assertEqual(en("Generate (生成)"), "Generate")
        self.assertEqual(bi("Generate (生成)"), "Generate (生成)")
        user_text = "My transcript: 这是我的声音。 (保持原样)"
        self.assertEqual(en(user_text), user_text)
        self.assertEqual(en("Chinese (中文)"), "Chinese")
        with self.assertRaises(ValueError):
            translator("invalid")

    def test_lightweight_launcher_does_not_import_model(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import qwen_tts_local.launcher; assert 'torch' not in sys.modules; assert 'qwen_tts' not in sys.modules",
            ],
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(
    sys.platform == "linux"
    and hasattr(os, "pidfd_open")
    and hasattr(signal, "pidfd_send_signal"),
    "Linux pidfd required",
)
class LauncherIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = Settings(
            checkpoint="never-loaded", state_dir=str(self.root), stop_wait_seconds=1
        )

    def tearDown(self):
        self.temp.cleanup()

    def held_child(self):
        """A real launcher child waiting at its handshake, before any model import."""
        lock = launcher.acquire_lock(self.root)
        read, write = os.pipe()
        command = [
            sys.executable,
            "-m",
            "qwen_tts_local.launcher",
            "_serve",
            "--instance",
            "test-instance",
            "--snapshot",
            str(self.root / "unused.json"),
            "--lock-fd",
            str(lock.fileno()),
            "--ready-fd",
            str(read),
        ]
        child = subprocess.Popen(
            command,
            pass_fds=(lock.fileno(), read),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        os.close(read)
        record = dict(
            instance="test-instance",
            identity=launcher.identity(child.pid),
            url=self.settings.url,
        )
        launcher.atomic_json(self.root / "run.json", record)
        # Simulate launcher exit: the child's inherited lock must still protect startup.
        lock.close()
        return child, write, record

    def cleanup_child(self, child, write):
        os.close(write)
        if child.poll() is None:
            child.terminate()  # only our owned handshake fixture, never a model
        child.communicate(timeout=5)

    def test_inherited_lock_prevents_duplicate_and_orphan_stop_works(self):
        child, write, record = self.held_child()
        try:
            with patch.object(launcher.subprocess, "Popen") as spawn:
                self.assertIn("Already running", launcher.start(self.settings))
                spawn.assert_not_called()
            self.assertIn("Running PID", launcher.status(self.settings))
            self.assertIn("Stopped verified", launcher.stop(self.settings))
            child.wait(timeout=5)
            self.assertFalse((self.root / "run.json").exists())
        finally:
            self.cleanup_child(child, write)

    def test_changed_start_time_refuses_signal(self):
        child, write, record = self.held_child()
        try:
            record["identity"]["start_ticks"] = "0"
            launcher.atomic_json(self.root / "run.json", record)
            with patch.object(launcher.signal, "pidfd_send_signal") as send:
                self.assertIn("no signal sent", launcher.stop(self.settings))
                send.assert_not_called()
            self.assertIsNone(child.poll())
        finally:
            self.cleanup_child(child, write)

    def test_changed_command_and_instance_refused(self):
        child, write, record = self.held_child()
        try:
            for field in ("argv", "boot_id", "exe"):
                corrupt = json.loads(json.dumps(record))
                if field == "argv":
                    corrupt["identity"][field][-1] = "different"
                else:
                    corrupt["identity"][field] = "different"
                with self.subTest(field=field), self.assertRaises(ProcessLookupError):
                    launcher.verified_pidfd(corrupt)
            record["instance"] = "wrong"
            with self.assertRaises(ValueError):
                launcher.verified_pidfd(record)
        finally:
            self.cleanup_child(child, write)

    def test_missing_record_and_unrelated_process_not_signalled(self):
        with patch.object(launcher.signal, "pidfd_send_signal") as send:
            self.assertIn("no signal sent", launcher.stop(self.settings))
            launcher.atomic_json(
                self.root / "run.json",
                dict(instance="unrelated", identity=launcher.identity(os.getpid())),
            )
            with self.assertRaises(ValueError):
                launcher.stop(self.settings)
            send.assert_not_called()

    def test_occupied_port_never_spawns(self):
        import socket

        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            settings = replace(self.settings, port=occupied.getsockname()[1])
            with (
                patch.object(launcher.subprocess, "Popen") as spawn,
                self.assertRaises(OSError),
            ):
                launcher.start(settings)
            spawn.assert_not_called()

    def test_parent_handshake_eof_exits_without_importing_model(self):
        child, write, record = self.held_child()
        os.close(write)
        out, err = child.communicate(timeout=5)
        self.assertEqual(child.returncode, 1, err)
        self.assertNotIn(b"Torch", out + err)
        with launcher.acquire_lock(self.root):
            pass

    def test_start_record_failure_releases_child_before_model_import(self):
        children = []
        real_spawn = subprocess.Popen
        real_save = launcher.atomic_json

        def spawn(*a, **kw):
            child = real_spawn(*a, **kw)
            children.append(child)
            return child

        def save(path, value):
            if path.name == "run.json":
                raise OSError("simulated disk write failure")
            return real_save(path, value)

        with (
            patch.dict(os.environ, {"PYTHONPATH": str(ROOT)}),
            patch.object(launcher.socket, "socket"),
            patch.object(launcher.subprocess, "Popen", side_effect=spawn),
            patch.object(launcher, "atomic_json", side_effect=save),
        ):
            with self.assertRaisesRegex(OSError, "disk write"):
                launcher.start(self.settings, False)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].wait(timeout=5), 1)
        self.assertFalse((self.root / "run.json").exists())
        with launcher.acquire_lock(self.root):
            pass

    def test_force_uses_same_verified_handle_only(self):
        record = dict(instance="fixture", identity={"pid": 123})
        launcher.atomic_json(self.root / "run.json", record)
        fd = os.open(os.devnull, os.O_RDONLY)
        with (
            patch.object(launcher, "verified_pidfd", return_value=fd),
            patch.object(
                launcher.select, "select", side_effect=[([], [], []), ([fd], [], [])]
            ),
            patch.object(launcher.signal, "pidfd_send_signal") as send,
        ):
            self.assertIn("Stopped verified", launcher.stop(self.settings, force=True))
            self.assertEqual(
                send.call_args_list,
                [
                    unittest.mock.call(fd, signal.SIGTERM),
                    unittest.mock.call(fd, signal.SIGKILL),
                ],
            )

    def test_default_stop_never_escalates_on_timeout(self):
        launcher.atomic_json(
            self.root / "run.json", dict(instance="fixture", identity={"pid": 123})
        )
        fd = os.open(os.devnull, os.O_RDONLY)
        with (
            patch.object(launcher, "verified_pidfd", return_value=fd),
            patch.object(launcher.select, "select", return_value=([], [], [])),
            patch.object(launcher.signal, "pidfd_send_signal") as send,
        ):
            self.assertIn("Still stopping", launcher.stop(self.settings))
            send.assert_called_once_with(fd, signal.SIGTERM)
        self.assertTrue((self.root / "run.json").exists())

    def test_start_ready_opens_only_local_url_after_identity_check(self):
        child = Mock(pid=123)
        child.poll.return_value = None
        response = Mock()
        response.__enter__ = Mock(return_value=types.SimpleNamespace(status=200))
        response.__exit__ = Mock(return_value=False)
        fd = os.open(os.devnull, os.O_RDONLY)
        with (
            patch.object(launcher.socket, "socket"),
            patch.object(launcher.subprocess, "Popen", return_value=child) as spawn,
            patch.object(launcher, "identity", return_value={"pid": 123}),
            patch.object(launcher, "verified_pidfd", return_value=fd),
            patch.object(launcher.urllib.request, "build_opener") as http,
            patch.object(launcher.webbrowser, "open") as browser,
        ):
            http.return_value.open.return_value = response
            self.assertIn("Ready", launcher.start(self.settings))
            browser.assert_called_once_with(self.settings.url)
            self.assertEqual(
                spawn.call_args.args[0][1:4],
                ["-m", "qwen_tts_local.launcher", "_serve"],
            )
            self.assertTrue(spawn.call_args.kwargs["start_new_session"])
            self.assertEqual(len(spawn.call_args.kwargs["pass_fds"]), 2)


if __name__ == "__main__":
    unittest.main()
