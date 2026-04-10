import os
import tempfile
import unittest
from pathlib import Path

from generator import AlphaGenerator
from pipeline import AlphaFactory
from alpha_factory_cli import _acquire_singleton_lock, _release_singleton_lock


class GeneratorHypothesisModeTests(unittest.TestCase):
    def test_hypothesis_mode_generates_structural_candidates(self):
        gen = AlphaGenerator(generation_mode="hypothesis_driven")
        batch = gen.generate_batch(n=12, use_rag=False)
        self.assertGreater(len(batch), 0)
        for cand in batch:
            expr = cand.expression.replace(" ", "")
            self.assertIn("rank(", expr)
            self.assertIn("group_neutralize(", expr)
            self.assertTrue(expr.startswith("group_neutralize(ts_decay_linear("))

    def test_pipeline_wires_generator_mode_from_env(self):
        old_mode = os.getenv("GENERATOR_MODE")
        os.environ["GENERATOR_MODE"] = "hypothesis_driven"
        try:
            factory = AlphaFactory(email="dummy", password="dummy")
            self.assertEqual(factory.generator.generation_mode, "hypothesis_driven")
            factory.tracker.close()
        finally:
            if old_mode is None:
                os.environ.pop("GENERATOR_MODE", None)
            else:
                os.environ["GENERATOR_MODE"] = old_mode

    def test_singleton_lock_prevents_duplicate_local_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = os.path.join(tmp, "runner.lock")
            fd1 = _acquire_singleton_lock(lock_path=Path(lock_path))
            self.assertIsNotNone(fd1)
            fd2 = _acquire_singleton_lock(lock_path=Path(lock_path))
            self.assertIsNone(fd2)
            _release_singleton_lock(fd1, Path(lock_path))
            fd3 = _acquire_singleton_lock(lock_path=Path(lock_path))
            self.assertIsNotNone(fd3)
            _release_singleton_lock(fd3, Path(lock_path))


if __name__ == "__main__":
    unittest.main()
