# Local launcher and preferences / 本地启动器与偏好设置

This fork includes an optional, lightweight `qwen-tts-local` command. It imports
neither Torch nor the model for configuration, status or stop operations. The
launcher uses **Linux pidfds** for precise process identity; Linux kernel 5.3+
and Python 3.10+ are required. Paths are configurable and use XDG defaults, with
no machine-specific installation paths. On other platforms use `qwen-tts-demo`
directly; the launcher refuses rather than falling back to unsafe PID-only stops.

本分支提供可选的轻量 `qwen-tts-local` 命令，配置、状态和停止操作不导入 Torch 或模型。
启动器使用 Linux pidfd 精确识别进程，需要 Linux 5.3+ 和 Python 3.10+。路径可配置，
默认遵循 XDG 规范。其他平台请直接使用 `qwen-tts-demo`；不会退化为仅按 PID 停止进程。

## First start / 首次启动

```sh
pip install -e '.[long-text]'
qwen-tts-local init
# Edit the printed JSON file: set checkpoint to an existing local model directory
# or a model ID already present in your Hugging Face cache.
qwen-tts-local start
qwen-tts-local status
qwen-tts-local stop
```

For a separate configuration use `qwen-tts-local --config ./local.json init`
and `qwen-tts-local --config ./local.json start`. `init` never overwrites a file.
Explicit `./` or `../` checkpoint paths and relative state/project directories
resolve beside the configuration file; model IDs remain unchanged. A desktop
launcher may invoke the same `start` or `stop` command using the executable in
your own environment. No desktop files, autostart entries or services are installed.

先创建配置文件并设置已有模型路径或已缓存模型 ID，再启动。独立配置可通过 `--config`
指定。`init` 不覆盖文件。相对目录以配置文件所在目录为基准。桌面快捷方式可调用同一命令，
但本工具不会安装桌面文件、自动启动项或系统服务。

## Configuration / 配置

| JSON key | Default | Meaning / 含义 |
|---|---|---|
| `checkpoint` | empty | Required existing local path or cached model ID / 必填模型路径或缓存 ID |
| `ui_language` | `en` | English-only (`en`) or existing English/Chinese (`bilingual`) / 英文或中英双语 |
| `state_dir` | XDG state directory | Lock, process identity, settings snapshots and append-only `gui.log` / 锁、进程身份、配置快照与日志 |
| `projects_dir` | XDG data directory | Saved projects and clips / 项目与音频 |
| `port` | `7867` | Local TCP port; binding is always `127.0.0.1` / 仅本机端口 |
| `device` | `cuda:0` | One CUDA device or `cpu` / 单个 CUDA 设备或 CPU |
| `dtype` | `bfloat16` | `bfloat16`, `float16` or `float32` / 数据类型 |
| `attention` | `sdpa` | `sdpa`, `eager` or installed `flash_attention_2` / 注意力实现 |
| `threads` / `interop_threads` | `2` / `1` | Process-local Torch thread counts / 进程内 Torch 线程数 |
| `gpu_memory_fraction` | `0.7` | CUDA allocator fraction; ignored on CPU / CUDA 分配器比例，CPU 下忽略 |
| `max_new_tokens` | `256` | Simple-tab generation budget / 简单合成选项卡令牌预算 |
| `section_max_new_tokens` | `256` | Long-text section budget / 长文本每段令牌预算 |
| `offline` | `true` | Child's offline model/cache mode / 子进程离线缓存模式 |
| `open_browser` | `true` | Open the local page after readiness; `start --no-browser` overrides / 就绪后打开页面 |
| `startup_wait_seconds` | `120` | Return “still starting” on expiry; never terminate for timeout / 等待超时只报告仍在启动 |
| `stop_wait_seconds` | `10` | Graceful-stop wait before reporting pending / 正常停止等待时间 |

The raw demo also accepts `--ui-language en|bilingual`, `--threads`,
`--interop-threads`, `--gpu-memory-fraction`, `--attention`,
`--max-new-tokens` and `--long-text-max-new-tokens`. Its UI default remains
bilingual; its bind default is now localhost. Language selection changes only
application text, not available synthesis languages or user-entered scripts.
Historical project error details and third-party exceptions are preserved rather
than rewriting user data. Thread/allocator settings apply before model loading.
Changing these preferences requires a normal stop/start; no live mutation occurs.

原始演示命令也支持上述界面、线程、GPU 比例、注意力和令牌参数；界面默认仍为双语，
绑定地址默认改为本机。界面语言不会删除合成语言或改写用户文本。历史项目错误和第三方
异常保持原样。设置在模型加载前生效，更改后需要正常停止并重新启动，不会修改运行中的实例。

## Process lifetime and safeguards / 进程生命周期与保护

- Startup holds a nonblocking file lock. The model child inherits that lock and
  keeps it for its lifetime, even after the launcher exits. An occupied port is
  refused without touching its owner. Repeated starts never load another model.
- Before importing Qwen, the child waits for the parent to record its identity:
  PID, boot ID, process start ticks, user ID, executable and exact argument list
  including an instance nonce. If the parent disappears before this handshake,
  the child exits without importing the model.
- Stop opens a kernel pidfd **before** verifying this identity. Signals target
  that handle, not a reused numeric PID. Only that exact model process receives
  SIGTERM. If needed, explicit `stop --force` sends SIGKILL to the same handle
  after the grace period. There is no process-name matching or broad kill.
- A stale/mismatched record never authorizes a signal. Malformed records fail
  closed; inspect them manually. Use one state directory per instance, and stop
  the old instance before changing `state_dir`. Do not remove a held lock file.
- Stop is immediate application shutdown, distinct from the GUI's **Stop after
  current section**, which saves the current clip before pausing the project.
  Interrupted sections can be resumed after restart.
- The optional custom **resource watchdog is disabled/not installed**: there is
  no host-RAM/PSI threshold that refuses or kills a generation. This is application
  behavior only. Linux OOM handling, permissions, firewall, authentication and
  other OS protections are unchanged. The separate, configurable CUDA allocator
  fraction remains active and can still cause a CUDA allocation error.
- No autostart, telemetry, public sharing, global environment mutation or system
  package changes are performed. `offline=false` explicitly allows normal model
  downloads in the child; the default is offline. Settings snapshots/logs remain
  in the state directory after abnormal exits for diagnosis, and are not uploaded.

启动锁由模型子进程持有，启动器退出后仍可防止重复加载。停止前通过 pidfd 验证精确身份，
不会按进程名称或过期 PID 批量终止。自定义资源监控未启用，不根据主机内存或 PSI 自动
拒绝或终止生成；这不改变 Linux OOM、权限、防火墙等系统保护。GPU 分配器比例是独立选项。
本工具不会自动启动、公开服务、上传数据或修改全局环境与系统软件。

## Verification limits / 验证范围

Tests use stub models and short handshake-only subprocesses, not a second Qwen
model. They cover configuration validation, both interface languages, preservation
of Chinese synthesis/text, configurable token budgets, inherited-lock duplicate
protection, orphan stop, stale/reused identities, occupied ports and packaging.
The launcher has not been used to launch a real model during this change. The
patched Torch/torchaudio integration gap described in [long-text projects](long-text-projects.md)
still applies. Only load trusted self-created `.pt` voice profiles.

测试使用模拟模型与仅等待握手的短进程，不启动第二个 Qwen 模型。此次没有通过新启动器
运行真实模型，修补后 Torch/torchaudio 的端到端验证仍待完成。仅加载可信、自行创建的音色文件。
