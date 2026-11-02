# SPDX-License-Identifier: Apache-2.0
"""Optional bilingual long-text workflow for the upstream Base demo."""

from pathlib import Path
from dataclasses import asdict
import os, json
from urllib.parse import quote
import gradio as gr
import numpy as np
import soundfile as sf
import torch
from ... import VoiceClonePromptItem
from .longtext_core import Projects
from .longtext_preview import TEMPLATE, CSS, JS, LABELS


def ended_naturally(result, eos):
    seq = getattr(result, "sequences", None)
    ids = eos if isinstance(eos, (list, tuple)) else [eos]
    return bool(
        seq is not None
        and seq.shape[0] == 1
        and seq.shape[-1] > 0
        and int(seq[0, -1].item()) in ids
    )


def add_long_text_ui(demo, tts, root, generation_kwargs=None):
    projects = Projects(Path(root))

    def synthesize(p, s, dest):
        folder = projects.path(p["id"])
        voice = folder / "voice.pt"
        if not voice.exists():
            items = tts.create_voice_clone_prompt(
                ref_audio=str(folder / p["reference"]),
                ref_text=p["transcript"],
                x_vector_only_mode=False,
            )
            tmp = folder / "voice.pt.tmp"
            torch.save({"items": [asdict(it) for it in items]}, tmp)
            os.replace(tmp, voice)
        payload = torch.load(voice, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("items"), list)
            or len(payload["items"]) != 1
        ):
            raise ValueError(
                "Use a single-reference voice file created by this Qwen GUI. (请使用本 Qwen 界面创建的单参考音色文件。)"
            )
        items = [VoiceClonePromptItem(**d) for d in payload["items"]]
        talker = tts.model.talker
        original = talker.generate
        natural = False

        def observed(*args, **kwargs):
            nonlocal natural
            result = original(*args, **kwargs)
            eos = kwargs.get(
                "eos_token_id", tts.model.config.talker_config.codec_eos_token_id
            )
            natural = ended_naturally(result, eos)
            return result

        # The shared single-model Gradio queue prevents overlapping model calls.
        talker.generate = observed
        try:
            wavs, sr = tts.generate_voice_clone(
                text=s["text"],
                language=p["language"],
                voice_clone_prompt=items,
                **{**(generation_kwargs or {}), "max_new_tokens": 256},
            )
        finally:
            talker.generate = original
        audio = np.asarray(wavs[0])
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError("Model returned invalid audio. (模型返回了无效音频。)")
        sf.write(str(dest), audio, sr, subtype="PCM_16")
        return natural

    cached_urls = {}

    def preview(p, busy=False):
        snapshot = projects.preview_snapshot(p)
        snapshot["busy"] = busy
        snapshot["labels"] = LABELS["bilingual"]
        for row in snapshot["rows"]:
            row["url"] = None
            if row["start"] is not None:
                key = (p["id"], row["clip"])
                if key not in cached_urls:
                    file = projects.path(p["id"]) / row["clip"]
                    projects.space(file.stat().st_size)
                    cached = timeline.move_resource_to_block_cache(file)
                    cached_urls[key] = "/gradio_api/file=" + quote(
                        str(cached), safe="/"
                    )
                row["url"] = cached_urls[key]
        return json.dumps(snapshot, ensure_ascii=False)

    def render(p, message, busy=False):
        rows = [
            [
                i + 1,
                x["text"],
                LABELS["bilingual"].get(x["status"], x["status"]),
                round(x.get("duration", 0), 2),
                x.get("error", ""),
            ]
            for i, x in enumerate(p["sections"])
        ]
        return (
            gr.update(choices=projects.list(), value=p["id"], interactive=not busy),
            rows,
            preview(p, busy),
            None,
            message,
            p["id"],
        )

    def fail(e):
        raise gr.Error("Operation failed (操作失败): " + str(e))

    with demo:
        loaded = gr.State(None)
        with gr.Tabs():
            with gr.Tab("Long Text Projects (长文本项目)"):
                gr.Markdown(
                    "Create a project, then Generate / Resume. The preview grows as consecutive sections finish. Click a marker or text row to highlight a whole section for editing and regeneration. Projects and clips save locally. (创建项目后点击生成或继续。连续段落完成时预览会扩展。点击标记或文本行以选择整个段落进行编辑和重新生成。项目与音频保存在本地。)"
                )
                with gr.Row():
                    title = gr.Textbox(label="Project name (项目名称)")
                    language = gr.Dropdown(
                        choices=[
                            (en + " (" + zh + ")", en)
                            for en, zh in [
                                ("Auto", "自动"),
                                ("Chinese", "中文"),
                                ("English", "英语"),
                                ("Japanese", "日语"),
                                ("Korean", "韩语"),
                                ("German", "德语"),
                                ("French", "法语"),
                                ("Russian", "俄语"),
                                ("Portuguese", "葡萄牙语"),
                                ("Spanish", "西班牙语"),
                                ("Italian", "意大利语"),
                            ]
                        ],
                        value="English",
                        label="Language (fixed for this project) (语言（项目内固定）)",
                    )
                longtext = gr.Textbox(
                    label="Full text (完整文本)",
                    lines=8,
                    placeholder="Paste the script here. Paragraphs and sentence endings guide section boundaries. (在此粘贴文本，将优先按段落与句末拆分。)",
                )
                with gr.Row():
                    reference = gr.Audio(
                        label="Shared reference recording (共用参考音频)",
                        type="filepath",
                        sources=["upload"],
                    )
                    transcript = gr.Textbox(
                        label="Exact reference transcript (参考音频的准确文本)", lines=3
                    )
                    voicefile = gr.File(
                        label="Or a saved voice file (.pt); takes priority (或使用已保存的音色文件（.pt），优先使用此文件)",
                        file_types=[".pt"],
                        type="filepath",
                    )
                gr.Markdown(
                    "Only load trusted, self-created voice profiles. Voice files are PyTorch checkpoints, not ordinary audio; weights_only is not a guarantee that an untrusted checkpoint is safe. (仅加载可信且自行创建的音色文件。音色文件是 PyTorch 检查点，并非普通音频；weights_only 不能保证不可信检查点安全。)"
                )
                create = gr.Button(
                    "Create project and split text (创建项目并拆分文本)",
                    variant="primary",
                )
                gr.Markdown(
                    "Projects and copied references/profiles are saved in the configured project directory. Limit: 2 million input characters per project; available disk and memory still limit practical size. (项目及复制的参考音频或音色文件保存在配置的项目目录中。每个项目最多两百万输入字符，实际容量仍受磁盘和内存限制。)"
                )
                with gr.Row():
                    project = gr.Dropdown(
                        choices=projects.list(),
                        label="Saved project (已保存的项目)",
                        interactive=True,
                    )
                    refresh = gr.Button("Refresh project list (刷新项目列表)")
                    load = gr.Button("Load project (加载项目)")
                with gr.Row():
                    generate = gr.Button(
                        "Generate / Resume (生成 / 继续)", variant="primary"
                    )
                    stop = gr.Button("Stop after current section (当前段落完成后停止)")
                timeline = gr.HTML(
                    json.dumps(
                        dict(
                            rows=[], labels=LABELS["bilingual"], prefix_count=0, total=0
                        )
                    ),
                    html_template=TEMPLATE,
                    css_template=CSS,
                    js_on_load=JS,
                    label="Growing assembled preview (持续更新的合成预览)",
                    show_label=False,
                )
                with gr.Row():
                    export_prefix = gr.Button(
                        "Export completed audio so far (导出目前已完成的音频)"
                    )
                    assemble = gr.Button("Export all sections WAV (导出所有段落 WAV)")
                gr.Markdown(
                    "Export stops at the first unfinished section. (导出将在第一个未完成的段落前停止。)"
                )
                combined = gr.Audio(
                    label="Exported WAV snapshot — play or download (导出的 WAV 快照 — 播放或下载)",
                    type="filepath",
                    interactive=False,
                )
                status = gr.Textbox(
                    label="Project status (项目状态)", interactive=False
                )
                with gr.Accordion(
                    "Detailed section statuses (详细段落状态)", open=False
                ):
                    table = gr.Dataframe(
                        headers=[
                            "Section (段落)",
                            "Text (文本)",
                            "Status (状态)",
                            "Seconds (秒数)",
                            "Details (详情)",
                        ],
                        datatype=["number", "str", "str", "number", "str"],
                        interactive=False,
                        label="Sections in playback order (按播放顺序排列的段落)",
                        wrap=True,
                    )
                gr.Markdown(
                    "The growing preview plays existing section files in order; a small transition gap may occur. It never skips a missing section. Export joins the playable prefix or the complete project into one continuous WAV. Ready means a natural end token was observed, not that pronunciation or every word was reviewed. (动态预览按顺序播放已有音频，切换时可能有短暂停顿，且不会跳过缺失段落。导出会将已完成的连续部分或完整项目合并为一个连续 WAV。就绪表示观察到自然结束令牌，并不表示已检查发音或每个词。)"
                )
    outputs = [project, table, timeline, combined, status, loaded]

    def error_frame(e):
        return (
            gr.update(interactive=True),
            gr.skip(),
            gr.skip(),
            None,
            "Paused after error:  (出错后暂停：)" + str(e),
            gr.skip(),
        )

    def create_cb(title, text, lang, ref, trans, voice):
        try:
            return render(
                projects.create(title, text, lang, ref, trans, voice),
                "Project saved. Choose Generate / Resume. (项目已保存，请选择生成或继续。)",
            )
        except Exception as e:
            fail(e)

    def load_cb(pid):
        if not pid:
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "Choose a saved project, then Load project. (请选择已保存的项目，再点击加载项目。)",
                gr.skip(),
            )
        try:
            return render(
                projects.load(pid),
                "Project loaded. The completed prefix is ready to preview. (项目已加载，可预览从开头连续完成的部分。)",
            )
        except Exception as e:
            fail(e)

    def run_cb(pid):
        try:
            for p, message in projects.run(pid, synthesize):
                yield render(p, message, busy=True)
            yield render(p, message)
        except Exception as e:
            yield error_frame(e)

    def selection(pid, evt):
        d = evt._data
        if not pid or d.get("project_id") != pid:
            raise ValueError(
                "Load this project before editing its sections. (请先加载此项目再编辑段落。)"
            )
        projects.checked_selection(pid, d.get("section_id"), d.get("revision"))
        return d["section_id"], d.get("text", "")

    def edit_event(pid, evt: gr.EventData):
        try:
            sid, text = selection(pid, evt)
            return render(
                projects.edit(pid, sid, text),
                "Section text saved; regenerate it before playback can pass this section. (段落文本已保存，请重新生成后再继续播放此段落及后续内容。)",
            )
        except Exception as e:
            fail(e)

    def split_event(pid, evt: gr.EventData):
        try:
            sid, text = selection(pid, evt)
            projects.edit(pid, sid, text)
            return render(
                projects.split_section(pid, sid),
                "Selected section split. Other section IDs and recordings are retained. (所选段落已拆分，其他段落的标识和音频保持不变。)",
            )
        except Exception as e:
            fail(e)

    def regen_event(pid, evt: gr.EventData):
        try:
            sid, text = selection(pid, evt)
            projects.edit(pid, sid, text)
            for p, message in projects.run(pid, synthesize, only=sid):
                yield render(p, message, busy=True)
            yield render(p, message)
        except Exception as e:
            yield error_frame(e)

    def stop_cb(pid):
        try:
            return projects.stop(pid)
        except Exception as e:
            fail(e)

    def assemble_cb(pid):
        try:
            return (
                projects.assemble(pid),
                "Exported all sections in order. This WAV is a snapshot of the current section versions. (已按顺序导出所有段落，此 WAV 是当前段落版本的快照。)",
            )
        except Exception as e:
            fail(e)

    def prefix_cb(pid):
        try:
            return (
                projects.assemble(pid, prefix=True),
                "Exported the completed contiguous prefix only; later sections are not included. (仅导出从开头连续完成的部分，不包含其后的段落。)",
            )
        except Exception as e:
            fail(e)

    with demo:
        create.click(
            create_cb,
            [title, longtext, language, reference, transcript, voicefile],
            outputs,
        )
        refresh.click(lambda: gr.update(choices=projects.list()), outputs=project)
        load.click(load_cb, [project], outputs)
        generate.click(run_cb, [loaded], outputs)
        timeline.section_save(edit_event, [loaded], outputs)
        timeline.section_split(split_event, [loaded], outputs)
        timeline.section_regenerate(regen_event, [loaded], outputs)
        stop.click(stop_cb, [loaded], [status], queue=False)
        assemble.click(assemble_cb, [loaded], [combined, status])
        export_prefix.click(prefix_cb, [loaded], [combined, status])
    return projects
