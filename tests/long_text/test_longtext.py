# SPDX-License-Identifier: Apache-2.0
import json, struct, tempfile, unittest, wave
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qwen_tts/cli/long_text"))
from longtext_core import Projects, split_text, MAX_TEXT


def wav(path, value=10, sr=24000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<h", value) * 240)


class TestProjects(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = Projects(self.root / "projects")
        self.ref = self.root / "ref.wav"
        wav(self.ref)

    def tearDown(self):
        self.tmp.cleanup()

    def create(self, text="First sentence. Second sentence."):
        return self.db.create("Test", text, "English", str(self.ref), "Reference")

    def synth(self, p, s, path):
        wav(path, 1 if s["text"].startswith("First") else 2)
        return True

    def finish(self, p):
        return list(self.db.run(p["id"], self.synth))[-1][0]

    def test_split_preserves_text_order(self):
        text = "Hello world. Next line!\n" + ("a long phrase " * 80) + "完了。次の文。"
        parts = split_text(text)
        self.assertTrue(all(len(x) <= 140 for x in parts))
        self.assertEqual("".join(text.split()), "".join("".join(parts).split()))

    def test_sentence_boundaries(self):
        self.assertEqual(split_text("One. Two? Three!"), ["One.", "Two?", "Three!"])

    def test_extreme_word(self):
        self.assertEqual("".join(split_text("a" * 501)), "a" * 501)

    def test_empty_and_input_limit(self):
        for t in (" ", "x" * (MAX_TEXT + 1)):
            with self.assertRaises(ValueError):
                split_text(t)

    def test_order_and_assemble(self):
        p = self.finish(self.create())
        out = self.db.assemble(p["id"])
        with wave.open(out) as w:
            data = struct.unpack("<480h", w.readframes(480))
        self.assertEqual(data[:240], (1,) * 240)
        self.assertEqual(data[240:], (2,) * 240)

    def test_regeneration_keeps_other_clip(self):
        p = self.finish(self.create())
        old = p["sections"][1]["clip"]
        oldbytes = (self.db.path(p["id"]) / old).read_bytes()
        sid = p["sections"][0]["id"]
        self.db.edit(p["id"], sid, "Replacement.")
        p = list(self.db.run(p["id"], self.synth, only=sid))[-1][0]
        self.assertEqual(p["sections"][1]["clip"], old)
        self.assertEqual((self.db.path(p["id"]) / old).read_bytes(), oldbytes)

    def test_resume_skips_ready(self):
        p = self.finish(self.create())
        calls = []
        list(self.db.run(p["id"], lambda *x: calls.append(x)))
        self.assertEqual(calls, [])

    def test_interruption_recovery(self):
        p = self.create()
        p["sections"][0]["status"] = "generating"
        self.db.save(p)
        self.assertEqual(self.db.load(p["id"])["sections"][0]["status"], "pending")

    def test_missing_clip_recovery(self):
        p = self.finish(self.create())
        (self.db.path(p["id"]) / p["sections"][0]["clip"]).unlink()
        self.assertEqual(self.db.load(p["id"])["sections"][0]["status"], "error")

    def test_stop_and_resume(self):
        p = self.create()
        calls = []

        def synth(p, s, f):
            calls.append(s["id"])
            wav(f)
            self.db.stop(p["id"])
            return True

        p = list(self.db.run(p["id"], synth))[-1][0]
        self.assertEqual(len(calls), 1)
        self.assertEqual(p["sections"][1]["status"], "pending")
        p = self.finish(p)
        self.assertTrue(all(s["status"] == "ready" for s in p["sections"]))

    def test_error_preserves_other_clips(self):
        p = self.finish(self.create())
        original = p["sections"][0]["clip"]

        def fail(*a):
            raise RuntimeError("mock failure")

        p = list(self.db.run(p["id"], fail, only=p["sections"][0]["id"]))[-1][0]
        self.assertEqual(p["sections"][0]["clip"], original)
        self.assertIn("mock failure", p["sections"][0]["error"])

    def test_format_mismatch(self):
        p = self.finish(self.create())
        wav(self.db.path(p["id"]) / p["sections"][1]["clip"], sr=16000)
        with self.assertRaisesRegex(ValueError, "sample rates"):
            self.db.assemble(p["id"])
        self.assertFalse((self.db.path(p["id"]) / "assembled.wav.tmp").exists())

    def test_unknown_end_blocks_assembly(self):
        p = self.create("Short.")

        def synth(p, s, f):
            wav(f)
            return False

        p = list(self.db.run(p["id"], synth))[-1][0]
        self.assertEqual(p["sections"][0]["status"], "needs_review")
        with self.assertRaises(ValueError):
            self.db.assemble(p["id"])

    def test_cap_retry_splits(self):
        p = self.create("word " * 24)

        def synth(p, s, f):
            wav(f)
            return len(s["text"]) < 70

        p = list(self.db.run(p["id"], synth))[-1][0]
        self.assertGreater(len(p["sections"]), 1)
        self.assertTrue(all(x["status"] == "ready" for x in p["sections"]))

    def test_disk_refusal(self):
        with patch(
            "longtext_core.shutil.disk_usage", return_value=type("D", (), {"free": 1})()
        ):
            with self.assertRaisesRegex(OSError, "free disk"):
                self.create()

    def test_malformed_manifest_and_path(self):
        with self.assertRaises(ValueError):
            self.db.load("../escape")
        p = self.create()
        p["sections"][0]["clip"] = "../../escape"
        self.db.save(p)
        with self.assertRaises(ValueError):
            self.db.load(p["id"])

    def test_atomic_manifest_previous_survives(self):
        p = self.create()
        before = (self.db.path(p["id"]) / "manifest.json").read_bytes()
        with patch("longtext_core.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.db.save(p)
        self.assertEqual((self.db.path(p["id"]) / "manifest.json").read_bytes(), before)

    def test_restart_mid_section_preserves_finished_clips(self):
        p = self.create()
        gen = self.db.run(p["id"], self.synth)
        next(gen)
        next(gen)
        next(gen)  # first saved, second marked generating
        ready = self.db.load(p["id"])["sections"][0]["clip"]
        gen.close()
        restarted = Projects(self.db.root)
        p = restarted.load(p["id"])
        self.assertEqual(p["sections"][1]["status"], "pending")
        p = list(restarted.run(p["id"], self.synth))[-1][0]
        self.assertEqual(p["sections"][0]["clip"], ready)
        self.assertTrue(all(x["status"] == "ready" for x in p["sections"]))

    def test_stop_request_during_blocked_inference(self):
        import threading

        p = self.create()
        entered = threading.Event()
        release = threading.Event()
        calls = []
        results = []

        def synth(p, s, f):
            calls.append(s["id"])
            entered.set()
            release.wait(3)
            wav(f)
            return True

        worker = threading.Thread(
            target=lambda: results.extend(self.db.run(p["id"], synth))
        )
        worker.start()
        self.assertTrue(entered.wait(2))
        self.db.stop(p["id"])
        self.assertTrue(worker.is_alive())
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(calls), 1)
        self.assertEqual(results[-1][0]["sections"][1]["status"], "pending")

    def test_edit_invalidates_only_selected_and_assembly(self):
        p = self.finish(self.create())
        assembled = Path(self.db.assemble(p["id"]))
        p = self.db.load(p["id"])
        self.assertIn("assembly", p)
        second = dict(p["sections"][1])
        p = self.db.edit(p["id"], p["sections"][0]["id"], "Changed text.")
        self.assertNotIn("assembly", p)
        self.assertIsNone(p["sections"][0]["clip"])
        self.assertEqual(p["sections"][1], second)
        self.assertTrue(
            assembled.exists()
        )  # archived output preserved, no longer current
        with self.assertRaises(ValueError):
            self.db.assemble(p["id"])

    def test_regeneration_invalidates_assembly_and_versions_export(self):
        p = self.finish(self.create())
        old = self.db.assemble(p["id"])
        p = list(self.db.run(p["id"], self.synth, only=p["sections"][0]["id"]))[-1][0]
        self.assertNotIn("assembly", self.db.load(p["id"]))
        new = self.db.assemble(p["id"])
        self.assertNotEqual(old, new)
        self.assertTrue(Path(old).exists())
        self.assertEqual(self.db.load(p["id"])["assembly"]["file"], Path(new).name)

    def test_disk_write_failure_mid_project_resume(self):
        p = self.create()
        calls = []

        def synth(p, s, f):
            calls.append(s["id"])
            if len(calls) == 2:
                raise OSError("No space left on device")
            wav(f)
            return True

        p = list(self.db.run(p["id"], synth))[-1][0]
        clip = p["sections"][0]["clip"]
        self.assertEqual(p["sections"][1]["status"], "error")
        self.assertIn("No space", p["sections"][1]["error"])
        p = self.finish(p)
        self.assertEqual(p["sections"][0]["clip"], clip)
        self.assertEqual(p["sections"][1]["status"], "ready")

    def test_assembly_disk_write_failure_retains_prior_export(self):
        p = self.finish(self.create())
        old = Path(self.db.assemble(p["id"]))
        data = old.read_bytes()
        with patch(
            "longtext_core.os.replace", side_effect=OSError("disk write failed")
        ):
            with self.assertRaises(OSError):
                self.db.assemble(p["id"])
        self.assertEqual(old.read_bytes(), data)
        self.assertFalse((self.db.path(p["id"]) / "assembled.wav.tmp").exists())

    def test_sample_format_mismatch(self):
        p = self.finish(self.create())
        f = self.db.path(p["id"]) / p["sections"][1]["clip"]
        with wave.open(str(f), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x00" * 480)
        with self.assertRaisesRegex(ValueError, "sample rates/formats"):
            self.db.assemble(p["id"])

    def test_prefix_grows_and_never_skips_failed_gap(self):
        p = self.create("First. Second. Third.")
        gen = self.db.run(p["id"], self.synth)
        next(gen)
        p, _ = next(gen)
        snap = self.db.preview_snapshot(p)
        self.assertEqual(snap["prefix_count"], 1)
        self.assertEqual(snap["rows"][0]["start"], 0)
        self.assertIsNone(snap["rows"][1]["start"])
        gen.close()
        p = self.finish(p)
        p["sections"][1]["status"] = "error"
        self.db.save(p)
        snap = self.db.preview_snapshot(p)
        self.assertEqual(snap["prefix_count"], 1)
        self.assertIsNone(snap["rows"][2]["start"])
        output = self.db.assemble(p["id"], prefix=True)
        with wave.open(output) as w:
            self.assertEqual(w.getnframes(), 240)
        with self.assertRaises(ValueError):
            self.db.assemble(p["id"])

    def test_prefix_empty_at_initial_gap(self):
        p = self.finish(self.create())
        p["sections"][0]["status"] = "needs_review"
        self.db.save(p)
        self.assertEqual(self.db.preview_snapshot(p)["prefix_count"], 0)
        with self.assertRaises(ValueError):
            self.db.assemble(p["id"], prefix=True)

    def test_timeline_ids_remain_stable_when_durations_change(self):
        p = self.finish(self.create())
        before = self.db.preview_snapshot(p)
        sid = p["sections"][1]["id"]
        revision = self.db.section_revision(p["sections"][1])

        def longer(p, s, f):
            with wave.open(str(f), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(b"\x00" * 960)
            return True

        p = list(self.db.run(p["id"], longer, only=p["sections"][0]["id"]))[-1][0]
        after = self.db.preview_snapshot(p)
        self.assertEqual(after["rows"][1]["id"], sid)
        self.assertEqual(after["rows"][1]["revision"], revision)
        self.assertEqual(after["rows"][1]["start"], 0.02)
        self.assertEqual(before["rows"][1]["start"], 0.01)

    def test_stale_selection_rejected(self):
        p = self.finish(self.create())
        s = p["sections"][0]
        revision = self.db.section_revision(s)
        self.db.edit(p["id"], s["id"], "Changed.")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.db.checked_selection(p["id"], s["id"], revision)

    def test_preview_uses_actual_audio_duration_not_stale_manifest(self):
        p = self.finish(self.create())
        p["sections"][0]["duration"] = 999
        self.assertEqual(self.db.preview_snapshot(p)["rows"][0]["duration"], 0.01)

    def test_language_persisted(self):
        p = self.create()
        p["language"] = "Chinese"
        self.db.save(p)
        self.assertEqual(self.db.load(p["id"])["language"], "Chinese")


if __name__ == "__main__":
    unittest.main()
