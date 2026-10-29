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
        self.assertEqual(rel.llr("s_behave", True), 0.0)
        self.assertIsNone(rel.combine(OutcomeSignals(s_behave=1.0, s_judge=1.0)))

    def test_verifier_overrides(self):
        rel = SourceReliability()
        self.assertEqual(rel.combine(OutcomeSignals(s_gt=0.0, s_behave=1.0)), 0.0)

    def test_always_yes_source_is_ignored_even_when_most_outcomes_succeed(self):
        # 90% of outcomes succeed, so "always yes" agrees with the verifier 90% of the time.
        # It still cannot tell success from failure, so its vote must carry no weight.
        rel = SourceReliability()
        for i in range(100):
            truth = 0.0 if i % 10 == 0 else 1.0
            rel.observe(OutcomeSignals(s_gt=truth, s_behave=1.0, s_judge=truth))
        self.assertEqual(rel.llr("s_behave", True), 0.0)
        self.assertGreater(rel.llr("s_judge", True), 1.0)
        self.assertLess(rel.llr("s_judge", False), -1.0)
        # Its "yes" alone is no evidence, so the step is not learned from.
        self.assertIsNone(rel.combine(OutcomeSignals(s_behave=1.0)))

    def test_constant_sources_carry_no_weight_with_few_and_unbalanced_audits(self):
        # 9 verified successes and 1 failure: a Beta(1, 1) prior centred on 0.5 would give an
        # always-yes source spec = 1/3 and a positive vote.
        for succ, fail in ((9, 1), (1, 9), (5, 5), (1, 1), (30, 1)):
            for says in (1.0, 0.0):
                rel = SourceReliability()
                for truth in [1.0] * succ + [0.0] * fail:
                    rel.observe(OutcomeSignals(s_gt=truth, s_behave=says), ["t"])
                for doc in (None, "t"):
                    for vote in (True, False):
                        self.assertEqual(rel.llr("s_behave", vote, doc), 0.0, (succ, fail, says))

    def test_informative_source_still_counts_with_few_audits(self):
        rel = SourceReliability()
        for truth in [1.0] * 9 + [0.0]:
            rel.observe(OutcomeSignals(s_gt=truth, s_judge=truth))
        self.assertGreater(rel.llr("s_judge", True), 0.0)
        self.assertLess(rel.llr("s_judge", False), 0.0)

    def test_learns_to_ignore_sycophantic_source(self):
        rel = SourceReliability()
        for truth in [1.0, 0.0] * 20:
            rel.observe(OutcomeSignals(s_gt=truth, s_behave=1.0, s_judge=truth))
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
        self.assertGreater(rel.llr("s_behave", True), 1.0)
        self.assertLess(rel.llr("s_behave", True, "poison"), 0.5)

        self.assertLess(rel.combine(OutcomeSignals(s_behave=1.0, s_judge=0.0), "poison"), 0.1)
        self.assertGreater(rel.combine(OutcomeSignals(s_behave=1.0), "good"), 0.9)

    def test_global_only_mode_ignores_documents(self):
        rel = SourceReliability(doc_strength=None)
        rel.observe(OutcomeSignals(s_gt=0.0, s_behave=1.0), ["poison"])
        self.assertEqual(rel.doc_counts, {})
        self.assertEqual(rel.rates("s_behave", "poison"), rel.rates("s_behave"))

    def test_round_trip(self):
        rel = SourceReliability()
        rel.observe(OutcomeSignals(s_gt=1.0, s_judge=1.0), ["d"])
        clone = SourceReliability.from_dict(rel.to_dict())
        self.assertEqual(clone.counts, rel.counts)
        self.assertEqual(clone.doc_counts, rel.doc_counts)
        self.assertEqual(clone.base, rel.base)


class TestChangeDetector(unittest.TestCase):
    def test_no_change_on_stable_history(self):
        c = Candidate(id="d", content="", recent_outcomes=[1.0] * 8)
        self.assertFalse(ChangeDetector(window=4, history="soft").detect(c))

    def test_resets_on_shift(self):
        c = Candidate(id="d", content="", alpha=9.0, beta=1.0, A=5.0, B=1.0)
        c.recent_outcomes = [1.0] * 4 + [0.0] * 4
        det = ChangeDetector(window=4, history="soft")
        self.assertTrue(det.check(c))
        self.assertEqual((c.alpha, c.beta), (1.0, 5.0))
        self.assertEqual((c.A, c.B), (5.0, 1.0))  # long-term counters kept by default
        self.assertEqual(c.recent_outcomes, [0.0] * 4)
        self.assertEqual(det.resets, 1)

    def test_optional_long_term_reset(self):
        c = Candidate(id="d", content="", A=5.0, B=1.0)
        c.recent_outcomes = [1.0] * 4 + [0.0] * 4
        ChangeDetector(window=4, reset_long_term=True, history="soft").check(c)
        self.assertEqual((c.A, c.B), (1.0, 2.0))

    def test_threshold_is_quantized_by_window(self):
        # Window means move in steps of 0.25: a 0.5 shift does not reach 0.6.
        c = Candidate(id="d", content="", recent_outcomes=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        self.assertFalse(ChangeDetector(window=4, threshold=0.6, history="soft").detect(c))


class TestStatisticalChangeDetector(unittest.TestCase):
    def test_noisy_but_stationary_outcomes_rarely_reset(self):
        # A tool that works for about half of the requests, with no change at all.
        import random

        rng = random.Random(0)
        fixed = ChangeDetector(window=4, threshold=0.6, history="soft")
        test = ChangeDetector(history="soft")
        for _ in range(2000):
            c = Candidate(id="d", content="")
            c.recent_outcomes = [float(rng.random() < 0.5) for _ in range(16)]
            fixed.check(c)
            c.recent_outcomes = [float(rng.random() < 0.5) for _ in range(16)]
            test.check(c)
        self.assertGreater(fixed.resets, 100)
        self.assertLess(test.resets, 40)

    def test_detects_clear_flip(self):
        c = Candidate(id="d", content="")
        c.recent_outcomes = [1.0] * 8 + [0.0] * 8
        self.assertTrue(ChangeDetector(history="soft").check(c))

    def test_checks_cluster_counters(self):
        c = Candidate(id="d", content="")
        c.recent_outcomes = [1.0] * 16  # globally stable
        c.cluster_counters = {
            "site": {"alpha": 9.0, "beta": 1.0, "recent_outcomes": [1.0] * 8 + [0.0] * 8}
        }
        det = ChangeDetector(history="soft")
        self.assertTrue(det.check(c, "site"))
        self.assertEqual(c.cluster_counters["site"]["beta"], 9.0)
        self.assertFalse(det.check(c))


class TestVerifiedChangeDetector(unittest.TestCase):
    def test_tests_verified_outcomes_not_soft_targets(self):
        # Soft targets look stable (unverified successes at 0.81), verified outcomes flip.
        c = Candidate(id="d", content="", alpha=9.0, beta=1.0)
        c.recent_outcomes = [0.81] * 16
        det = ChangeDetector()
        for y in [1.0] * 8 + [0.0] * 8:
            det.observe("d", y > 0.5)
        self.assertTrue(det.check(c))
        self.assertEqual((c.alpha, c.beta), (1.0, 9.0))
        self.assertEqual(c.recent_outcomes, [0.81] * 16)  # soft history left alone
        self.assertFalse(det.check(c))  # the change is not detected twice

    def test_no_verified_history_no_reset(self):
        c = Candidate(id="d", content="", recent_outcomes=[1.0] * 8 + [0.0] * 8)
        self.assertFalse(ChangeDetector().check(c))

    def test_false_resets_on_stationary_verified_outcomes(self):
        # Testing after every verified outcome with overlapping windows: count false alarms.
        import random

        rng = random.Random(1)
        det = ChangeDetector()
        for tool in range(8):
            c = Candidate(id=f"t{tool}", content="")
            for _ in range(500):
                det.observe(c.id, rng.random() < 0.7)
                det.check(c)
        # 4,000 tests; repeated testing inflates the 1% level, but resets stay rare.
        self.assertLess(det.resets, 40)

    def test_layer_detects_defection_hidden_by_unverified_success(self):
        store = CandidateStore()
        det = ChangeDetector()
        layer = ReputationLayer(store, gamma=1.0, change_detector=det)
        for step in range(200):
            res = layer.rescore({"d": 1.0}, top_k=1, explore=False, now=float(step))
            verified = step % 5 == 0
            truth = 1.0 if step < 100 else 0.0
            layer.record_feedback(
                res.response_id,
                s_gt=truth if verified else None,
                s_behave=1.0,
                s_judge=1.0,
                now=float(step),
            )
        self.assertGreaterEqual(det.resets, 1)


class TestClusterShrinkage(unittest.TestCase):
    def test_cluster_evidence_overrides_global_reputation(self):
        store = CandidateStore()
        layer = ReputationLayer(store, weights=(0.0, 1.0, 0.0, 0.0), gamma=1.0)
        # "a" works on site x and fails on site y; "b" is the reverse.
        for step in range(20):
            for site, good in (("x", "a"), ("y", "b")):
                res = layer.rescore(
                    {"a": 1.0, "b": 1.0}, top_k=2, explore=False, cluster_id=site, now=float(step)
                )
                for doc in ("a", "b"):
                    r = layer.rescore(
                        {doc: 1.0}, top_k=1, explore=False, cluster_id=site, now=float(step)
                    )
                    layer.record_feedback(
                        r.response_id, s_gt=1.0 if doc == good else 0.0, now=float(step)
                    )
                store.pop_pending(res.response_id)
        top_x = layer.rescore({"a": 1.0, "b": 1.0}, top_k=1, explore=False, cluster_id="x").results[
            0
        ][0]
        top_y = layer.rescore({"a": 1.0, "b": 1.0}, top_k=1, explore=False, cluster_id="y").results[
            0
        ][0]
        self.assertEqual((top_x, top_y), ("a", "b"))


class TestConfidentGreedy(unittest.TestCase):
    def _scores(self, layer):
        return {
            tuple(layer.rescore({"a": 1.0, "b": 1.0}, top_k=2, explore=True, epsilon=0.0).results)
            for _ in range(10)
        }

    def test_stops_exploring_once_the_leader_is_clear(self):
        store = CandidateStore()
        store.add_candidate(Candidate(id="a", content="", alpha=40.0, beta=1.0))
        store.add_candidate(Candidate(id="b", content="", alpha=1.0, beta=40.0))
        layer = ReputationLayer(store, confident_greedy=0.95)
        self.assertEqual(len(self._scores(layer)), 1)  # no random exploration term

    def test_keeps_exploring_while_uncertain(self):
        store = CandidateStore()
        store.add_candidate(Candidate(id="a", content="", alpha=2.0, beta=1.0))
        store.add_candidate(Candidate(id="b", content="", alpha=1.0, beta=2.0))
        layer = ReputationLayer(store, confident_greedy=0.95)
        self.assertGreater(len(self._scores(layer)), 1)

    def test_off_by_default(self):
        store = CandidateStore()
        store.add_candidate(Candidate(id="a", content="", alpha=40.0, beta=1.0))
        store.add_candidate(Candidate(id="b", content="", alpha=1.0, beta=40.0))
        self.assertGreater(len(self._scores(ReputationLayer(store))), 1)


class TestStickToVerified(unittest.TestCase):
    def _layer(self):
        store = CandidateStore()
        layer = ReputationLayer(store, gamma=1.0, stick_to_verified=True)
        return store, layer

    def _choose(self, layer, doc_scores, site="x"):
        return layer.rescore(doc_scores, top_k=1, explore=True, cluster_id=site)

    def test_verified_success_locks_in_until_a_verified_failure(self):
        _, layer = self._layer()
        sims = {"a": 0.2, "b": 1.0}
        r = layer.rescore({"a": 0.2}, top_k=1, explore=False, cluster_id="x")
        layer.record_feedback(r.response_id, s_gt=1.0)
        picks = {self._choose(layer, sims).results[0][0] for _ in range(20)}
        self.assertEqual(picks, {"a"})  # despite b's higher relevance, and no exploration

        r = self._choose(layer, sims)
        layer.record_feedback(r.response_id, s_gt=0.0)
        self.assertEqual(self._choose(layer, {"a": 0.2, "b": 1.0}).results[0][0], "b")

    def test_unverified_success_does_not_lock_in(self):
        _, layer = self._layer()
        r = layer.rescore({"a": 0.2}, top_k=1, explore=False, cluster_id="x")
        layer.record_feedback(r.response_id, s_behave=1.0, s_judge=1.0)
        picks = {self._choose(layer, {"a": 0.2, "b": 1.0}).results[0][0] for _ in range(20)}
        self.assertIn("b", picks)

    def test_lock_is_per_context(self):
        _, layer = self._layer()
        r = layer.rescore({"a": 0.2}, top_k=1, explore=False, cluster_id="x")
        layer.record_feedback(r.response_id, s_gt=1.0)
        picks = {
            self._choose(layer, {"a": 0.2, "b": 1.0}, site="y").results[0][0] for _ in range(20)
        }
        self.assertIn("b", picks)


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
