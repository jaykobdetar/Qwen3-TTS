# SPDX-License-Identifier: Apache-2.0
"""Construct actual Gradio blocks with a model stub, never import torch/Qwen or launch a server."""

import ast, tempfile, types, sys
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gradio as gr
import numpy as np

# Load the feature with heavy imports excluded, leaving model functions uncalled.
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "qwen_tts/cli"))
p = root / "qwen_tts/cli/long_text/longtext_ui.py"
tree = ast.parse(p.read_text())
tree.body = [
    n
    for n in tree.body
    if not (isinstance(n, ast.Import) and any(a.name == "torch" for a in n.names))
    and not (isinstance(n, ast.ImportFrom) and n.level == 3)
]
ns = {"__package__": "long_text"}
exec(compile(tree, str(p), "exec"), ns)
assert ns["ended_naturally"](
    types.SimpleNamespace(sequences=np.array([[1, 2, 99]])), 99
)
assert not ns["ended_naturally"](
    types.SimpleNamespace(sequences=np.array([[1, 2, 3]])), 99
)
assert not ns["ended_naturally"](types.SimpleNamespace(), 99)
assert not ns["ended_naturally"](types.SimpleNamespace(sequences=np.empty((1, 0))), 99)
original = root / "qwen_tts/cli/demo.py"
names = (
    "_title_case_display",
    "_build_choices_and_map",
    "_detect_model_kind",
    "build_demo",
)
body = [
    n
    for n in ast.parse(original.read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name in names
]
module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *body,
    ],
    type_ignores=[],
)


# Exercise the actual optional build_demo integration without importing the model package.
class FeatureImport(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == "long_text.longtext_ui":
            node.level = 0
        return node


module = FeatureImport().visit(module)
sys.modules["long_text.longtext_ui"] = types.SimpleNamespace(
    add_long_text_ui=ns["add_long_text_ui"]
)
from qwen_tts_local.language import translator

base = {"gr": gr, "translator": translator}
exec(compile(ast.fix_missing_locations(module), str(original), "exec"), base)
fake = types.SimpleNamespace(
    model=types.SimpleNamespace(
        tts_model_type="base",
        get_supported_languages=lambda: ["auto", "chinese", "english"],
        get_supported_speakers=lambda: [],
    )
)
with tempfile.TemporaryDirectory() as d:
    demo = base["build_demo"](fake, "stub-model", {"max_new_tokens": 256}, Path(d))
    demo.queue(default_concurrency_limit=1)
    for fn in demo.fns.values():
        if fn.queue:
            demo._queue.create_event_queue_for_fn(fn)
    assert list(demo._queue.event_queue_per_concurrency_id) == ["qwen-long-text-model"]
    assert (
        demo._queue.event_queue_per_concurrency_id[
            "qwen-long-text-model"
        ].concurrency_limit
        == 1
    )
    config = demo.get_config_file()
    labels = [c["props"].get("label") for c in config["components"]]
    # All feature labels, placeholders, buttons and table headers are bilingual.
    feature_ids = set(
        range(
            next(
                c["id"]
                for c in config["components"]
                if c["props"].get("label", "").startswith("Long Text Projects")
            ),
            max(c["id"] for c in config["components"]) + 1,
        )
    )
    for c in config["components"]:
        if c["id"] not in feature_ids:
            continue
        props = c["props"]
        for key in ("label", "placeholder"):
            if props.get(key):
                assert any("\u3400" <= ch <= "\u9fff" for ch in props[key]), (
                    key,
                    props[key],
                )
        if c["type"] in ("button", "markdown") and props.get("value"):
            assert any("\u3400" <= ch <= "\u9fff" for ch in props["value"]), props[
                "value"
            ]
    assert any(
        "Long Text Projects" in str(x) and "长文本项目" in str(x) for x in labels
    )
    assert any("Clone & Generate" in str(x) for x in labels) and any(
        "Save / Load Voice" in str(x) for x in labels
    )
    stop = next(fn for fn in demo.fns.values() if fn.name == "stop_cb")
    assert stop.queue is False
    assert all(
        fn.concurrency_id == "qwen-long-text-model" and fn.concurrency_limit == 1
        for fn in demo.fns.values()
    )
    # Exercise project creation, growing-prefix outputs and direct stable-ID events.
    ref = Path(d) / "voice.pt"
    ref.write_bytes(b"not loaded by creation")
    create = next(fn.fn for fn in demo.fns.values() if fn.name == "create_cb")
    result = create("UI test", "Hello. Another line.", "English", None, "", str(ref))
    pid = result[0]["value"]
    import json

    view = json.loads(result[2])
    assert len(view["rows"]) == 2 and view["prefix_count"] == 0
    assert result[5] == pid
    load = next(fn.fn for fn in demo.fns.values() if fn.name == "load_cb")
    assert len(load(pid)[1]) == 2
    assert (
        "Choose a saved project" in load(None)[4]
    )  # missing project handled without exception
    ns["torch"] = types.SimpleNamespace(
        load=lambda *a, **k: {
            "items": [
                dict(
                    ref_code=None,
                    ref_spk_embedding=[1],
                    x_vector_only_mode=True,
                    icl_mode=False,
                    ref_text=None,
                )
            ]
        }
    )
    ns["VoiceClonePromptItem"] = lambda **kw: types.SimpleNamespace(**kw)

    class Talker:
        def generate(self, **kwargs):
            return types.SimpleNamespace(sequences=np.array([[1, 2, 99]]))

    talker = Talker()
    original = talker.generate
    fake.model.talker = talker
    fake.model.config = types.SimpleNamespace(
        talker_config=types.SimpleNamespace(codec_eos_token_id=99)
    )

    def voice_clone(**kwargs):
        assert kwargs["max_new_tokens"] == 256
        fake.model.talker.generate(eos_token_id=99)
        return [np.ones(240, dtype=np.float32) * 0.1], 24000

    fake.generate_voice_clone = voice_clone
    run = next(fn.fn for fn in demo.fns.values() if fn.name == "run_cb")
    frames = list(run(pid))
    views = [json.loads(f[2]) for f in frames]
    assert any(v["prefix_count"] == 1 for v in views) and views[-1]["prefix_count"] == 2
    assert views[0]["busy"] and not views[-1]["busy"]
    assert talker.generate == original
    assert len({v["rows"][0]["url"] for v in views if v["prefix_count"]}) == 1
    assert all(
        f[3] is None for f in frames
    )  # old exported snapshot cleared, custom player remains independent
    row = views[-1]["rows"][0]
    event = types.SimpleNamespace(
        _data=dict(
            project_id=pid,
            section_id=row["id"],
            revision=row["revision"],
            text="Changed.",
        )
    )
    regenerate = next(fn.fn for fn in demo.fns.values() if fn.name == "regen_event")
    reg = list(regenerate(pid, event))
    latest = json.loads(reg[-1][2])
    assert latest["rows"][0]["id"] == row["id"]
    assert latest["rows"][0]["text"] == "Changed."
    stale = list(regenerate(pid, event))
    assert "changed" in stale[-1][4]
    assembly = next(fn.fn for fn in demo.fns.values() if fn.name == "assemble_cb")
    assert Path(assembly(pid)[0]).is_file()
    print(
        "Real Gradio construction, single queue/stop bypass, loaded-project state, custom events, growing prefix, cached URLs, stale selection refusal, EOS capture and assembly passed. No model/server loaded."
    )

# The English preference changes application text, never script content or the
# synthesis language values. Exercise the configurable section budget as well.
with tempfile.TemporaryDirectory() as d:
    demo = base["build_demo"](fake, "stub-model", {}, Path(d), "en", 128)
    cfg = demo.get_config_file()
    for component in cfg["components"]:
        props = component["props"]
        for key in ("label", "placeholder", "html_template"):
            if isinstance(props.get(key), str):
                assert not any("\u3400" <= ch <= "\u9fff" for ch in props[key]), (
                    key,
                    props[key],
                )
        if component["type"] in ("button", "markdown") and isinstance(
            props.get("value"), str
        ):
            assert not any("\u3400" <= ch <= "\u9fff" for ch in props["value"]), props[
                "value"
            ]
    assert any("Chinese" in str(c["props"].get("choices")) for c in cfg["components"])
    ref = Path(d) / "voice.pt"
    ref.write_bytes(b"stub")
    create = next(fn.fn for fn in demo.fns.values() if fn.name == "create_cb")
    result = create("Test", "保留用户的中文。", "Chinese", None, "", str(ref))
    view = json.loads(result[2])
    assert view["rows"][0]["text"] == "保留用户的中文。"
    assert not any("\u3400" <= ch <= "\u9fff" for ch in view["labels"]["heading"])
    calls = []

    def custom_budget(**kwargs):
        calls.append(kwargs)
        fake.model.talker.generate(eos_token_id=99)
        return [np.ones(240, dtype=np.float32) * 0.1], 24000

    fake.generate_voice_clone = custom_budget
    run = next(fn.fn for fn in demo.fns.values() if fn.name == "run_cb")
    list(run(result[5]))
    assert calls[0]["max_new_tokens"] == 128 and calls[0]["language"] == "Chinese"
    assert calls[0]["text"] == "保留用户的中文。"
    try:
        create("Test", "", "English", None, "", str(ref))
    except gr.Error as e:
        assert not any("\u3400" <= ch <= "\u9fff" for ch in str(e))
    else:
        raise AssertionError("Expected empty-text error")
    print(
        "English UI, preserved Chinese synthesis/text, translated errors and configurable 128-token section budget passed."
    )
