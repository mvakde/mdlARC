import json
import tempfile
import unittest
from pathlib import Path

from server import Catalog


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_json(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_discovers_raw_tasks_without_aggregate_duplicates(self):
        pair = {"input": [[0]], "output": [[1]]}
        self.write_json("ARC/raw/training/abc.json", {"train": [pair], "test": [pair]})
        self.write_json("ARC/all.json", {"abc": {"train": [pair]}})

        catalog = Catalog(self.root)

        self.assertEqual(catalog.datasets[0]["taskCount"], 1)
        self.assertEqual(catalog.datasets[0]["tasks"][0]["key"], "training/abc")

    def test_pages_list_style_tasks_in_chunks_of_twelve(self):
        pairs = [{"input": [[index % 10]], "output": [[0]]} for index in range(25)]
        self.write_json("RE-ARC/tasks/generated.json", pairs)
        catalog = Catalog(self.root)

        payload = catalog.page("RE-ARC", "generated", 1)

        self.assertEqual(payload["pairCount"], 25)
        self.assertEqual(payload["pageCount"], 3)
        self.assertEqual(payload["start"], 12)
        self.assertEqual(payload["end"], 24)
        self.assertEqual(len(payload["pairs"]), 12)
        self.assertTrue(all(pair["split"] == "generated" for pair in payload["pairs"]))

    def test_clamps_pages_and_keeps_train_test_labels(self):
        pair = {"input": [[0]], "output": [[1]]}
        self.write_json("Concept/raw/Group/task.json", {"train": [pair], "test": [pair, pair]})
        catalog = Catalog(self.root)

        payload = catalog.page("Concept", "Group/task", 99)

        self.assertEqual(payload["page"], 0)
        self.assertEqual([item["split"] for item in payload["pairs"]], ["train", "test", "test"])


if __name__ == "__main__":
    unittest.main()
