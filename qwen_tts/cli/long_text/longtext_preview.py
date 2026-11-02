# SPDX-License-Identifier: Apache-2.0
"""Small dependency-free section timeline; UI strings are isolated for upstream localization."""

from pathlib import Path

LABELS = {
    "en": dict(
        noProject="Create or load a project to start.",
        heading="Growing assembled preview",
        help="Click a numbered marker or text row to select a whole section. Selection is aligned to section boundaries, not arbitrary waveform slices.",
        play="Play preview",
        pause="Pause",
        section="Section",
        choose="Select a section",
        notInPrefix="Not in playable prefix yet",
        prefix="sections playable from the beginning",
        gap="Playback stops at the first pending, failed or unconfirmed section; later clips are never skipped into the preview.",
        complete="All sections are in the preview.",
        fromSelection="Play from selected section",
        save="Save section text",
        regenerate="Regenerate highlighted section",
        split="Split highlighted section",
        reload="Reload selected text",
        editor="Highlighted section text",
        waiting="Reached the completed prefix. Waiting for the next contiguous section.",
        invalidated="The playing version is no longer current. Preview paused; select a current section to continue.",
        replaced="This section changed. Preview paused with its position preserved where possible.",
        conflict="This section changed elsewhere. Reload selected text before editing.",
        pressPlay="Press Play to continue playback.",
        audioError="This clip could not be loaded. Reload the project; do not skip the failed gap.",
    ),
    "zh": dict(
        noProject="创建或加载项目以开始。",
        heading="持续更新的合成预览",
        help="点击编号标记或文本行以选择整个段落。选择范围与段落边界对齐，不支持任意波形切片。",
        play="播放预览",
        pause="暂停",
        section="段落",
        choose="选择段落",
        notInPrefix="尚未进入可播放的连续部分",
        prefix="个段落可从开头连续播放",
        gap="播放将在第一个待处理、失败或未确认的段落处停止，不会跳过缺失段落。",
        complete="所有段落均已加入预览。",
        fromSelection="从所选段落播放",
        save="保存段落文本",
        regenerate="重新生成高亮段落",
        split="拆分高亮段落",
        reload="重新加载所选文本",
        editor="高亮段落文本",
        waiting="已播放完已完成的连续部分，正在等待下一段。",
        invalidated="正在播放的版本已过期。预览已暂停，请选择当前段落继续。",
        replaced="此段落已更新。预览已暂停，并尽可能保留播放位置。",
        conflict="此段落已在其他位置更改，请先重新加载所选文本。",
        pressPlay="请点击播放以继续。",
        audioError="无法加载此音频。请重新加载项目，不要跳过缺失段落。",
    ),
}

LABELS["en"].update(
    pending="pending",
    generating="generating",
    ready="ready",
    error="error",
    needs_review="needs review",
)
LABELS["zh"].update(
    pending="待处理",
    generating="生成中",
    ready="就绪",
    error="错误",
    needs_review="需要检查",
)
LABELS["bilingual"] = {
    key: value + " (" + LABELS["zh"][key] + ")" for key, value in LABELS["en"].items()
}

TEMPLATE = """<section class="lt-preview" aria-label="Growing assembled preview (持续更新的合成预览)">
<h3 data-label="heading"></h3><p data-label="help"></p><strong data-role="summary"></strong><p data-role="gap"></p>
<audio preload="metadata"></audio><div data-role="markers" class="lt-markers" aria-label="Section boundary markers (段落边界标记)"></div>
<div class="lt-controls"><button type="button" data-role="play"></button><span data-role="clock">0:00 / 0:00</span><button type="button" data-role="fromSelection" data-label="fromSelection"></button></div>
<input type="range" data-role="seek" aria-label="Position in completed prefix (已完成连续部分中的播放位置)" min="0" max="0" step="0.01" value="0">
<div data-role="notice" role="status" aria-live="polite"></div><div data-role="sections" class="lt-sections" aria-label="Select a whole section (选择整个段落)"></div>
<h4 data-role="selected"></h4><label data-label="editor"></label><textarea data-role="editor" aria-label="Highlighted section text (高亮段落文本)" rows="4"></textarea>
<div class="lt-controls"><button type="button" data-role="save" data-label="save"></button><button type="button" data-role="regenerate" data-label="regenerate"></button><button type="button" data-role="split" data-label="split"></button><button type="button" data-role="reload" data-label="reload"></button></div></section>"""
CSS = """.lt-preview{border:1px solid #667085;border-radius:12px;padding:16px;font:inherit;color:inherit;background:var(--block-background-fill)}
.lt-preview button{padding:9px 12px;border:1px solid #667085;border-radius:7px;background:var(--button-secondary-background-fill,#e8e8f0);color:inherit;cursor:pointer}
.lt-preview button.selected{background:#5145cd;color:white;border:2px solid #aaa2ff}.lt-preview button:focus-visible{outline:3px solid #be9eff;outline-offset:2px}.lt-preview button:disabled{opacity:.5;cursor:default}
.lt-markers{display:flex;gap:2px;min-height:42px;overflow-x:auto;margin:12px 0}.lt-markers button{min-width:36px;border-radius:3px}.lt-controls{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:8px 0}
.lt-preview input[type=range],.lt-preview textarea{width:100%;box-sizing:border-box}.lt-preview textarea{padding:10px;background:var(--input-background-fill);color:inherit;border:1px solid #667085;border-radius:7px}.lt-sections{display:grid;gap:5px;max-height:240px;overflow-y:auto;margin-top:12px}.lt-sections button{text-align:left;white-space:normal}.lt-preview [data-role=notice]{min-height:1.5em;color:var(--body-text-color)}"""
JS = Path(__file__).with_suffix(".js").read_text(encoding="utf-8")
