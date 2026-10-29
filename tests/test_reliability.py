import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rrl.drift import ChangeDetector
from rrl.feedback import OutcomeSignals
from rrl.layer import ReputationLayer
from rrl.reliability import SourceReliability
from rrl.store import Candidate, CandidateStore


class TestSourceReliability(unittest.TestCase):
    def test_unchecked_sources_carry_no_weight(self):
        rel = SourceReliability()
        self.assertEqual(rel.weight("s_behave"), 0.0)
        self.assertIsNone(rel.combine(OutcomeSignals(s_behave=1.0, s_judge=1.0)))

    def test_verifier_overrides(self):
        rel = SourceReliability()
        self.assertEqual(rel.combine(OutcomeSignals(s_gt=0.0, s_behave=1.0)), 0.0)

    def test_learns_to_ignore_sycophantic_source(self):
        rel = SourceReliability()
        # The user always accepts; the judge agrees with the verifier.
        for truth in [1.0, 0.0] * 20:
            rel.observe(OutcomeSignals(s_gt=truth, s_behave=1.0, s_judge=truth))
        self.assertAlmostEqual(rel.accuracy("s_behave"), 0.5, places=6)
        self.assertEqual(rel.weight("s_behave"), 0.0)
        self.assertGreater(rel.weight("s_judge"), 1.0)

        # Unverified event: user says yes, judge says no -> trust the judge.
        y = rel.combine(OutcomeSignals(s_behave=1.0, s_judge=0.0))
        self.assertIsNotNone(y)
        self.assertLess(y, 0.1)

    def test_catches_source_fooled_by_one_document(self):
        rel = SourceReliability()
        # The user is honest on "good" but always accepts answers built from "poison".
        for truth in [1.0, 0.0] * 20:
            rel.observe(OutcomeSignals(s_gt=truth, s_behave=truth, s_judge=truth), ["good"])
        for _ in range(10):
            rel.observe(OutcomeSignals(s_gt=0.0, s_behave=1.0, s_judge=0.0), ["poison"])
        self.assertGreater(rel.weight("s_behave"), 1.0)
        self.assertEqual(rel.weight("s_behave", "poison"), 0.0)

        unverified = OutcomeSignals(s_behave=1.0, s_judge=0.0)
        self.assertLess(rel.combine(unverified, "poison"), 0.1)
        self.assertGreater(rel.combine(OutcomeSignals(s_behave=1.0), "good"), 0.9)

    def test_round_trip(self):
        rel = SourceReliability()
        rel.observe(OutcomeSignals(s_gt=1.0, s_judge=1.0), ["d"])
        clone = SourceReliability.from_dict(rel.to_dict())
        self.assertEqual(clone.counts, rel.counts)
        self.assertEqual(clone.doc_counts, rel.doc_counts)


class TestChangeDetector(unittest.TestCase):
    def test_no_change_on_stable_history(self):
        c = Candidate(id="d", content="", recent_outcomes=[1.0] * 8)
        self.assertFalse(ChangeDetector(window=4).detect(c))

    def test_resets_on_shift(self):
        c = Candidate(id="d", content="", alpha=9.0, beta=1.0)
        c.recent_outcomes = [1.0] * 4 + [0.0] * 4
        self.assertTrue(ChangeDetector(window=4).check(c))
        self.assertEqual((c.alpha, c.beta), (1.0, 5.0))
        self.assertEqual(c.recent_outcomes, [0.0] * 4)


class TestLayerIntegration(unittest.TestCase):
    def test_skips_update_from_untrusted_sources(self):
        store = CandidateStore()
        layer = ReputationLayer(store, reliability=SourceReliability(), gamma=1.0)
        res = layer.rescore({"d": 1.0}, top_k=1, explore=False, now=0.0)
        layer.record_feedback(res.response_id, s_behave=1.0, now=1.0)
        cand = store.get_candidate("d")
        self.assertEqual((cand.alpha, cand.beta), (1.0, 1.0))

    def test_change_detector_reacts_to_flip(self):
        store = CandidateStore()
        layer = ReputationLayer(
            store, gamma=1.0, change_detector=ChangeDetector(window=4, threshold=0.6)
        )
        for step, outcome in enumerate([1.0] * 8 + [0.0] * 4):
            res = layer.rescore({"d": 1.0}, top_k=1, explore=False, now=float(step))
            layer.record_feedback(res.response_id, s_gt=outcome, now=float(step))
        cand = store.get_candidate("d")
        # Without the reset alpha would still hold the 8 early successes.
        self.assertLess(cand.alpha / (cand.alpha + cand.beta), 0.3)


if __name__ == "__main__":
    unittest.main()
