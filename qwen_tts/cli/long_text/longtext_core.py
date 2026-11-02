# SPDX-License-Identifier: Apache-2.0
"""Local resumable audio projects. No model imports or network access."""

import json, os, re, shutil, threading, uuid, wave, hashlib
from pathlib import Path

MAX_TEXT = 2_000_000
RESERVE = 128 * 1024 * 1024


def split_text(text, limit=140):
    text = text.strip()
    if not text:
        raise ValueError("Enter some text first. (请先输入文本。)")
    if len(text) > MAX_TEXT:
        raise ValueError(
            "Use separate projects for inputs over 2 million characters. (超过两百万字符的输入请拆分为多个项目。)"
        )
    chunks = []
    # Sentence/paragraph boundaries first; long sentences split at whitespace or character boundaries.
    for sentence in re.split(r"(?<=[.!?。！？])\s+|(?<=[。！？])|\n+", text):
        sentence = sentence.strip()
        while sentence:
            n = min(
                limit,
                70 if re.search("[\u3400-\u9fff\u3040-\u30ff]", sentence) else limit,
            )
            if len(sentence) <= n:
                chunks.append(sentence)
                break
            cut = sentence.rfind(" ", 0, n + 1)
            if cut < n // 3:
                cut = n
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
    return chunks


def section(text, depth=0):
    return dict(
        id=uuid.uuid4().hex,
        text=text,
        status="pending",
        clip=None,
        error="",
        depth=depth,
    )


class Projects:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stops = {}
        self.lock = threading.Lock()

    def path(self, pid):
        if not re.fullmatch("[0-9a-f]{32}", pid or ""):
            raise ValueError("Select a local project. (请选择本地项目。)")
        return self.root / pid

    def space(self, extra=0):
        if shutil.disk_usage(self.root).free < RESERVE + extra:
            raise OSError(
                "Not enough free disk space. Free space, then resume; existing clips are retained. (磁盘空间不足。请释放空间后继续；已有音频会保留。)"
            )

    @staticmethod
    def assembly_sources(p):
        return [[s["id"], s["text"], s.get("clip"), s["status"]] for s in p["sections"]]

    def save(self, p):
        if p.get("assembly", {}).get("sources") != self.assembly_sources(p):
            p.pop("assembly", None)
        self.space(65536)
        folder = self.path(p["id"])
        folder.mkdir(exist_ok=True)
        tmp = folder / "manifest.json.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(p, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, folder / "manifest.json")

    def load(self, pid):
        p = json.loads((self.path(pid) / "manifest.json").read_text(encoding="utf-8"))
        if p.get("id") != pid or p.get("version") != 1:
            raise ValueError("Unsupported project manifest. (不支持此项目清单。)")
        for s in p["sections"]:
            if not re.fullmatch("[0-9a-f]{32}", s["id"]):
                raise ValueError("Invalid section ID. (段落标识无效。)")
            if s.get("clip") and not re.fullmatch(
                r"[0-9a-f]{32}-[0-9a-f]{32}\.wav", s["clip"]
            ):
                raise ValueError("Invalid clip path. (音频路径无效。)")
            if s["status"] == "generating":
                s["status"] = "pending"
                s["error"] = (
                    "Interrupted previously; resume to retry. (上次生成已中断，请继续以重试。)"
                )
            if s["status"] == "ready":
                try:
                    with wave.open(str(self.path(pid) / s["clip"]), "rb") as w:
                        if w.getnframes() == 0:
                            raise ValueError("Empty clip (音频为空)")
                except Exception:
                    s["status"] = "error"
                    s["error"] = (
                        "Missing or invalid clip; regenerate this section. (音频缺失或无效，请重新生成此段落。)"
                    )
        if p.get("assembly", {}).get("sources") != self.assembly_sources(p):
            p.pop("assembly", None)
        return p

    def list(self):
        result = []
        for f in sorted(
            self.root.glob("*/manifest.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        ):
            try:
                p = self.load(f.parent.name)
                result.append((p["title"], p["id"]))
            except Exception:
                continue
        return result

    def create(
        self, title, text, language, reference=None, transcript="", profile=None
    ):
        chunks = split_text(text)
        self.space()
        if not profile and (not reference or not transcript.strip()):
            raise ValueError(
                "Supply a saved voice file, or reference audio and its transcript. (请提供已保存的音色文件，或参考音频及其准确文本。)"
            )
        p = dict(
            version=1,
            id=uuid.uuid4().hex,
            title=(title.strip() or text[:40])[:120],
            language=language,
            transcript=transcript.strip(),
            sections=[section(t) for t in chunks],
        )
        folder = self.path(p["id"])
        folder.mkdir()
        source = Path(profile or reference)
        self.space(source.stat().st_size)
        shutil.copyfile(
            source,
            folder / ("voice.pt" if profile else "reference" + source.suffix.lower()),
        )
        p["reference"] = None if profile else "reference" + source.suffix.lower()
        self.save(p)
        return p

    def stop(self, pid):
        self.path(pid)
        with self.lock:
            self.stops.setdefault(pid, threading.Event()).set()
        return "Stop requested. The current section will finish; completed clips are saved. (已请求停止。当前段落完成后停止，已完成音频会保存。)"

    def event(self, pid):
        with self.lock:
            return self.stops.setdefault(pid, threading.Event())

    def edit(self, pid, sid, text):
        p = self.load(pid)
        s = next(s for s in p["sections"] if s["id"] == sid)
        if not text.strip():
            raise ValueError("Section text cannot be empty. (段落文本不能为空。)")
        if len(text) > 1000:
            raise ValueError(
                "Keep an edited section under 1,000 characters; use Split selected for shorter parts. (编辑后的段落应少于一千字符；可拆分所选段落以缩短。)"
            )
        if text.strip() != s["text"]:
            s.update(
                text=text.strip(),
                status="pending",
                clip=None,
                error="Text changed; regenerate before assembling. (文本已更改，请先重新生成再合并。)",
            )
            self.save(p)
        return p

    def split_section(self, pid, sid):
        p = self.load(pid)
        i = next(i for i, s in enumerate(p["sections"]) if s["id"] == sid)
        s = p["sections"][i]
        parts = split_text(s["text"], max(10, len(s["text"]) // 2))
        if len(parts) < 2:
            raise ValueError(
                "Section is too short to split further. (此段落过短，无法继续拆分。)"
            )
        p["sections"][i : i + 1] = [section(t, s["depth"] + 1) for t in parts]
        self.save(p)
        return p

    def run(self, pid, synthesize, only=None):
        p = self.load(pid)
        event = self.event(pid)
        event.clear()
        i = 0
        while i < len(p["sections"]):
            s = p["sections"][i]
            if event.is_set():
                break
            if (only and s["id"] != only) or (not only and s["status"] == "ready"):
                i += 1
                continue
            old = dict(s)
            try:
                self.space(4 * 1024 * 1024)
                s.update(status="generating", error="")
                self.save(p)
                yield p, "Generating section  (正在生成段落)" + str(i + 1)
                clip = self.path(pid) / (s["id"] + "-" + uuid.uuid4().hex + ".wav")
                # synthesize writes a PCM WAV and returns True only if natural end-of-speech was observed.
                natural = synthesize(p, s, clip)
                with wave.open(str(clip), "rb") as w:
                    if (
                        not w.getnframes()
                        or w.getnchannels() != 1
                        or w.getsampwidth() != 2
                    ):
                        raise ValueError(
                            "Generated audio is empty or not mono PCM16. (生成的音频为空或不是单声道 PCM16。)"
                        )
                    duration = w.getnframes() / w.getframerate()
                s.update(
                    clip=clip.name,
                    status="ready" if natural else "needs_review",
                    duration=duration,
                    error=(
                        ""
                        if natural
                        else "No confirmed natural ending; possible token-limit truncation. Split shorter or regenerate. (未确认自然结束，可能因令牌上限截断。请拆短或重新生成。)"
                    ),
                )
                # Preserve the suspect clip on disk; automatically retry smaller parts only during bulk generation.
                if (
                    not natural
                    and not only
                    and len(s["text"]) > 40
                    and s["depth"] < 3
                    and not event.is_set()
                ):
                    parts = split_text(s["text"], max(20, len(s["text"]) // 2))
                    p["sections"][i : i + 1] = [
                        section(t, s["depth"] + 1) for t in parts
                    ]
                    self.save(p)
                    yield p, "Possible truncation: retrying smaller sections. (可能被截断，正在拆分为更短段落后重试。)"
                    continue
                self.save(p)
                yield p, "Saved section  (已保存段落)" + str(i + 1)
                if not natural:
                    break
            except Exception as e:
                # A failed replacement leaves the previous ready recording usable, with a visible error.
                s.clear()
                s.update(old)
                if old["status"] != "ready":
                    s["status"] = "error"
                s["error"] = "Generation failed (生成失败): " + str(e)
                try:
                    self.save(p)
                except OSError:
                    pass
                yield p, "Paused after error:  (出错后暂停：)" + str(e)
                return
            if only:
                break
            i += 1
        yield p, (
            "Stopped after current section. Resume when ready. (当前段落完成后已停止，可随时继续。)"
            if event.is_set()
            else "Generation pass finished. Review statuses before assembling. (本轮生成结束，请检查各段落状态后再合并。)"
        )

    @staticmethod
    def section_revision(s):
        return hashlib.sha256(
            json.dumps([s["id"], s["text"], s.get("clip")], ensure_ascii=False).encode()
        ).hexdigest()

    def checked_selection(self, pid, sid, revision):
        p = self.load(pid)
        s = next((x for x in p["sections"] if x["id"] == sid), None)
        if s is None or self.section_revision(s) != revision:
            raise ValueError(
                "This section changed. Reload the project and select it again before editing. (此段落已更改，请重新加载项目并再次选择后编辑。)"
            )
        return p, s

    def preview_snapshot(self, p):
        rows = []
        elapsed = 0.0
        blocked = False
        count = 0
        for i, s in enumerate(p["sections"]):
            duration = None
            start = None
            end = None
            if not blocked and s["status"] == "ready" and s.get("clip"):
                try:
                    with wave.open(str(self.path(p["id"]) / s["clip"]), "rb") as w:
                        if (
                            w.getnchannels() != 1
                            or w.getsampwidth() != 2
                            or w.getnframes() == 0
                        ):
                            raise ValueError("Invalid preview audio (预览音频无效)")
                        duration = w.getnframes() / w.getframerate()
                    start = elapsed
                    elapsed += duration
                    end = elapsed
                    count += 1
                except (OSError, wave.Error, ValueError):
                    blocked = True
            else:
                blocked = True
            rows.append(
                dict(
                    id=s["id"],
                    number=i + 1,
                    text=s["text"],
                    status=s["status"],
                    error=s.get("error", ""),
                    revision=self.section_revision(s),
                    clip=s.get("clip"),
                    start=start,
                    end=end,
                    duration=duration,
                )
            )
        return dict(
            project_id=p["id"],
            title=p["title"],
            rows=rows,
            prefix_count=count,
            total=elapsed,
        )

    def assemble(self, pid, prefix=False):
        p = self.load(pid)
        count = (
            self.preview_snapshot(p)["prefix_count"] if prefix else len(p["sections"])
        )
        rows = p["sections"][:count]
        if not rows:
            raise ValueError(
                "No completed contiguous prefix is available yet. (尚无从开头连续完成的音频。)"
            )
        if any(s["status"] != "ready" for s in rows):
            raise ValueError(
                "Every section must be ready. Resolve pending/error/possible-truncation sections first. (所有段落必须就绪，请先处理未完成、错误或可能截断的段落。)"
            )
        files = [self.path(pid) / s["clip"] for s in rows]
        total = sum(f.stat().st_size for f in files)
        self.space(total)
        if total > 3_900_000_000:
            raise ValueError(
                "Combined WAV would exceed the supported WAV size. Divide this project into smaller projects. (合并后的 WAV 将超出支持的大小，请拆分项目。)"
            )
        final = self.path(pid) / (
            ("preview-" if prefix else "assembled-") + uuid.uuid4().hex + ".wav"
        )
        tmp = self.path(pid) / "assembled.wav.tmp"
        try:
            with wave.open(str(tmp), "wb") as out:
                fmt = None
                for f in files:
                    with wave.open(str(f), "rb") as w:
                        current = (w.getnchannels(), w.getsampwidth(), w.getframerate())
                        if fmt is None:
                            fmt = current
                            out.setnchannels(fmt[0])
                            out.setsampwidth(fmt[1])
                            out.setframerate(fmt[2])
                        if fmt != current:
                            raise ValueError(
                                "Section sample rates/formats differ. Regenerate mismatching sections; no silent resampling. (段落采样率或格式不一致，请重新生成不匹配的段落；不会自动重采样。)"
                            )
                        while True:
                            block = w.readframes(65536)
                            if not block:
                                break
                            out.writeframesraw(block)
            os.replace(tmp, final)
        finally:
            tmp.unlink(missing_ok=True)
        if not prefix:
            p["assembly"] = {"file": final.name, "sources": self.assembly_sources(p)}
        self.save(p)
        return str(final)
