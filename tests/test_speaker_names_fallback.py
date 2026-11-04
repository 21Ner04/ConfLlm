import json
import tempfile
import unittest
from pathlib import Path

from speaker_names_fallback import write_fallback


class FallbackTests(unittest.TestCase):
    def test_preserves_source_and_creates_unresolved_speakers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "transcript.json"
            transcript = {
                "utterances": [
                    {"id": 0, "speaker": "speaker_1", "text": "Привет"},
                    {"id": 1, "speaker": "speaker_2", "text": "Здравствуйте"},
                    {"id": 2, "speaker": "unknown", "text": "Шум"},
                ]
            }
            source.write_text(
                json.dumps(transcript, ensure_ascii=False),
                encoding="utf-8",
            )
            before = source.read_bytes()
            result = write_fallback(source, root / "names", "Тестовое предупреждение")

            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(
                set(result["speakers"]), {"speaker_1", "speaker_2"}
            )
            for entry in result["speakers"].values():
                self.assertIsNone(entry["name"])
                self.assertEqual(entry["status"], "unresolved")
            self.assertEqual(result["resolution_status"], "failed_fallback")
            self.assertEqual(
                result["source_transcript"], str(source.resolve())
            )

    def test_keeps_existing_attempt_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "transcript.json"
            source.write_text(
                '{"utterances": [{"speaker": "speaker_1"}]}',
                encoding="utf-8",
            )
            out = root / "names"
            out.mkdir()
            log = out / "attempt_1.txt"
            log.write_text("Неудачный ответ", encoding="utf-8")

            write_fallback(source, out, "Ошибка имён")

            self.assertEqual(
                log.read_text(encoding="utf-8"), "Неудачный ответ"
            )
            saved = json.loads(
                (out / "speaker_names.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["warnings"], ["Ошибка имён"])


if __name__ == "__main__":
    unittest.main()
