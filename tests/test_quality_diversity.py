import unittest

from quality_diversity import QualityDiversityArchive, behavior_descriptor


class QualityDiversityTests(unittest.TestCase):
    def test_behavior_descriptor_extracts_structure(self):
        expr = "group_neutralize(ts_corr(close, volume, 20), industry)"
        desc = behavior_descriptor(expr)
        self.assertIn("g|c|", desc)
        self.assertTrue(desc.endswith("industry"))

    def test_novelty_decreases_after_seen(self):
        archive = QualityDiversityArchive()
        expr = "group_neutralize(ts_zscore(ts_corr(close, volume, 20), 10), industry)"
        first, desc = archive.novelty_score(expr)
        archive.maybe_update_archive(expr, quality=1.2, novelty=first, descriptor=desc)
        second, _ = archive.novelty_score(expr)
        self.assertGreater(first, second)


if __name__ == "__main__":
    unittest.main()
