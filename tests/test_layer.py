import unittest
import os
import sys
import tempfile
import subprocess
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rrl.store import Candidate, CandidateStore
from rrl.store_sqlite import SqliteCandidateStore
from rrl.layer import ReputationLayer
from rrl.feedback import OutcomeSignals


class TestLayerLogic(unittest.TestCase):
    def test_rescore_ranks_by_formula(self):
        store = CandidateStore()
        # Candidate 1: established, high alpha, C_robust = 5/7 = 0.714
        c1 = Candidate(id="c1", content="Doc 1", alpha=5.0, beta=2.0, A=3.0, B=2.0)
        # Candidate 2: established, low alpha, C_robust = 2/5 = 0.4
        c2 = Candidate(id="c2", content="Doc 2", alpha=2.0, beta=3.0, A=1.0, B=4.0)
        store.add_candidate(c1)
        store.add_candidate(c2)

        # Layer with weights: w_sim=0.0, w_c=1.0, w_p=0.0 (focus only on robust score)
        layer = ReputationLayer(store, weights=(0.0, 1.0, 0.0, 0.0), robust_estimator_mode="beta")

        sims = {"c1": 0.9, "c2": 0.1}
        res = layer.rescore(sims, top_k=2, explore=False)

        # Ranked list should have c1 first
        self.assertEqual(res.results[0][0], "c1")
        self.assertAlmostEqual(res.results[0][1], 5.0 / 7.0)
        self.assertEqual(res.results[1][0], "c2")
        self.assertAlmostEqual(res.results[1][1], 2.0 / 5.0)

    def test_record_feedback_updates_counters(self):
        store = CandidateStore()
        c1 = Candidate(id="c1", content="Doc 1", alpha=1.0, beta=1.0, A=1.0, B=1.0)
        store.add_candidate(c1)

        layer = ReputationLayer(store, weights=(0.5, 0.5, 0.0, 0.0))
        # Call rescore to populate pending credit shares
        res = layer.rescore({"c1": 1.0}, top_k=1, explore=False)
        response_id = res.response_id

        # Submit feedback: s_gt=1.0 (verified success)
        layer.record_feedback(response_id, s_gt=1.0)

        # Candidate counters should be updated
        updated_c1 = store.get_candidate("c1")
        self.assertGreater(updated_c1.alpha, 1.0)
        self.assertEqual(updated_c1.verified, 1.0)
        self.assertEqual(updated_c1.fooled, 0.0)

    def test_roundtrip_both_stores(self):
        # 1. In-memory store
        store_mem = CandidateStore()
        c_mem = Candidate(id="c", content="Test", alpha=1.0, beta=1.0, A=1.0, B=1.0)
        store_mem.add_candidate(c_mem)
        layer_mem = ReputationLayer(store_mem, weights=(0.5, 0.5, 0.0, 0.0))

        # 2. SQLite store
        db_file_fd, db_file_path = tempfile.mkstemp()
        try:
            store_sql = SqliteCandidateStore(db_path=db_file_path)
            c_sql = Candidate(id="c", content="Test", alpha=1.0, beta=1.0, A=1.0, B=1.0)
            store_sql.add_candidate(c_sql)
            layer_sql = ReputationLayer(store_sql, weights=(0.5, 0.5, 0.0, 0.0))

            # Rescore and record feedback on both
            res_mem = layer_mem.rescore({"c": 0.8}, top_k=1, explore=False)
            res_sql = layer_sql.rescore({"c": 0.8}, top_k=1, explore=False)

            self.assertAlmostEqual(res_mem.results[0][1], res_sql.results[0][1])

            layer_mem.record_feedback(res_mem.response_id, s_behave=0.9, s_gt=0.0)
            layer_sql.record_feedback(res_sql.response_id, s_behave=0.9, s_gt=0.0)

            updated_mem = store_mem.get_candidate("c")
            updated_sql = store_sql.get_candidate("c")

            self.assertAlmostEqual(updated_mem.alpha, updated_sql.alpha)
            self.assertAlmostEqual(updated_mem.beta, updated_sql.beta)
            self.assertAlmostEqual(updated_mem.A, updated_sql.A)
            self.assertAlmostEqual(updated_mem.B, updated_sql.B)
            self.assertEqual(updated_mem.fooled, updated_sql.fooled)
            self.assertEqual(updated_mem.verified, updated_sql.verified)

        finally:
            os.close(db_file_fd)
            if os.path.exists(db_file_path):
                os.unlink(db_file_path)

    def test_unknown_candidate_on_the_fly(self):
        store = CandidateStore()
        layer = ReputationLayer(store)

        # Document "unknown" is NOT in the store
        self.assertIsNone(store.get_candidate("unknown"))

        # rescore should track it on the fly
        res = layer.rescore({"unknown": 1.0}, top_k=1, explore=False)
        self.assertEqual(res.results[0][0], "unknown")

        # Now it should be in the store
        cand = store.get_candidate("unknown")
        self.assertIsNotNone(cand)
        self.assertEqual(cand.id, "unknown")
        self.assertEqual(cand.content, "")

    def test_lazy_imports_isolation(self):
        # Verify that sentence_transformers is NOT imported upon loading rrl / ReputationLayer
        cmd = (
            "import sys\n"
            "import rrl\n"
            "from rrl.layer import ReputationLayer\n"
            "assert 'sentence_transformers' not in sys.modules, 'sentence_transformers was eagerly imported!'\n"
        )
        res = subprocess.run(
            [sys.executable, "-c", cmd],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))}
        )
        self.assertEqual(res.returncode, 0, f"Bare import check failed: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
