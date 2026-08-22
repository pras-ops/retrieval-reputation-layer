"""
Regression tests for the four defects that made outcome feedback inert or backwards.

Each test fails on the pre-fix code. They are written against observable behaviour
rather than internals so a future refactor cannot quietly reintroduce the defect.
"""

import unittest
import warnings

from rrl.store import Candidate, CandidateStore
from rrl.layer import ReputationLayer
from rrl.feedback import (
    Attribution,
    OutcomeSignals,
    VerifierNoise,
    calculate_kappa,
    calculate_outcome,
    correct_outcome,
    update_counters,
)


def _observe(store, cid, y, t, gamma=1.0, unit=1.0, **kw):
    sig = OutcomeSignals(s_gt=y, **kw)
    update_counters(
        store,
        {cid: 1.0},
        calculate_outcome(sig, use_safeguards=True),
        current_timestamp=float(t),
        gamma=gamma,
        decay_unit_sec=unit,
        credit_smoothing=0.0,
        signals=sig,
    )


class TestClockBase(unittest.TestCase):
    """An explicit timestamp of 0.0 is a real time, not a missing one."""

    def test_zero_timestamp_is_not_treated_as_missing(self):
        c = Candidate(id="d", content="x", last_confirmed=0.0)
        self.assertEqual(c.last_confirmed, 0.0)
        self.assertEqual(c.decay_anchor, 0.0)

    def test_simulated_time_actually_decays(self):
        """
        Pre-fix, a candidate's anchor was silently overwritten with wall-clock time, so a
        simulation passing small step numbers produced a negative dt and decay never ran.
        """
        store = CandidateStore()
        store.add_candidate(Candidate(id="d", content="x", alpha=5.0, beta=1.0, last_confirmed=0.0))
        layer = ReputationLayer(store, gamma=0.5, decay_unit_sec=1.0, use_optimistic_prior=False)
        alpha, _, _, _ = store.get_candidate("d").effective_counters(
            now=3.0, gamma=0.5, decay_unit_sec=1.0
        )
        self.assertLess(alpha, 5.0)
        self.assertAlmostEqual(alpha, 1.0 + 4.0 * 0.5**3)
        # And the layer must agree with that view rather than using a wall-clock anchor.
        layer.rescore({"d": 1.0}, top_k=1, explore=False, now=3.0)


class TestReadOnlyDecay(unittest.TestCase):
    """Scoring a document must not change its stored reputation."""

    def test_repeated_reads_are_idempotent(self):
        store = CandidateStore()
        store.add_candidate(Candidate(id="d", content="x", alpha=6.0, beta=2.0, last_confirmed=0.0))
        layer = ReputationLayer(store, gamma=0.95, decay_unit_sec=1.0, use_optimistic_prior=False)
        for step in range(1, 40):
            layer.rescore({"d": 1.0}, top_k=1, explore=False, now=float(step))
        c = store.get_candidate("d")
        self.assertEqual(c.alpha, 6.0, "reads must not write decayed values back")
        self.assertEqual(c.beta, 2.0)

    def test_decay_depends_on_elapsed_time_not_read_count(self):
        """
        Pre-fix, decay was re-applied from a stale anchor on every read, compounding as
        gamma**sum(dt) instead of gamma**dt. One read and many reads must now agree.
        """
        store = CandidateStore()
        store.add_candidate(Candidate(id="d", content="x", alpha=6.0, beta=2.0, last_confirmed=0.0))
        layer = ReputationLayer(store, gamma=0.9, decay_unit_sec=1.0, use_optimistic_prior=False)

        for step in range(1, 10):
            layer.rescore({"d": 1.0}, top_k=1, explore=False, now=float(step))
        many_reads = store.get_candidate("d").effective_counters(10.0, 0.9, 1.0)[0]

        store2 = CandidateStore()
        store2.add_candidate(
            Candidate(id="d", content="x", alpha=6.0, beta=2.0, last_confirmed=0.0)
        )
        single_read = store2.get_candidate("d").effective_counters(10.0, 0.9, 1.0)[0]

        self.assertAlmostEqual(many_reads, single_read)


class TestFeedbackAnchor(unittest.TestCase):
    """Failures are observations too, so they must advance the decay anchor."""

    def test_failure_advances_the_anchor(self):
        store = CandidateStore()
        store.add_candidate(Candidate(id="d", content="x", last_confirmed=0.0))
        _observe(store, "d", 0.0, t=7.0)
        c = store.get_candidate("d")
        self.assertEqual(c.last_feedback, 7.0, "a failure is an observation")
        self.assertEqual(c.last_confirmed, 0.0, "but it is not a confirmation")
        self.assertEqual(c.decay_anchor, 7.0)

    def test_negative_evidence_is_not_immortal(self):
        """
        Pre-fix, a document that only ever failed kept its original anchor forever, so its
        penalty never aged while a successful document's reward decayed away. Positive and
        negative evidence must have the same half-life.
        """
        store = CandidateStore()
        store.add_candidate(Candidate(id="fail", content="x", last_confirmed=0.0))
        store.add_candidate(Candidate(id="pass", content="y", last_confirmed=0.0))
        for t in range(1, 6):
            _observe(store, "fail", 0.0, t=t)
            _observe(store, "pass", 1.0, t=t)

        gamma, unit, later = 0.5, 1.0, 25.0
        f = store.get_candidate("fail")
        p = store.get_candidate("pass")
        f_alpha, f_beta, _, _ = f.effective_counters(later, gamma, unit)
        p_alpha, p_beta, _, _ = p.effective_counters(later, gamma, unit)
        # Both should have relaxed most of the way back toward the Beta(1,1) prior.
        self.assertLess(f_beta - 1.0, 0.05 * (f.beta - 1.0))
        self.assertLess(p_alpha - 1.0, 0.05 * (p.alpha - 1.0))


class TestExplorationDoesNotDominate(unittest.TestCase):
    def test_warns_when_exploration_outweighs_relevance(self):
        store = CandidateStore()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ReputationLayer(store, weights=(0.20, 0.40, 0.10, 0.30))
        self.assertTrue(
            any("exploration will dominate" in str(w.message) for w in caught),
            "a config where w_explore > w_sim/2 must warn",
        )

    def test_thompson_mode_is_bounded_by_its_weight(self):
        """
        The legacy term added w_explore * sim * (sample + rarity), whose range exceeds the
        relevance term. Pure Thompson sampling perturbs the estimate it replaces, so the
        exploration contribution cannot exceed w_explore.
        """
        store = CandidateStore()
        store.add_candidate(Candidate(id="a", content="x"))
        layer = ReputationLayer(
            store, weights=(0.70, 0.20, 0.10, 0.10), gamma=1.0, exploration_mode="ts"
        )
        base = layer.rescore({"a": 1.0}, top_k=1, explore=False).results[0][1]
        for _ in range(200):
            s = layer.rescore({"a": 1.0}, top_k=1, explore=True).results[0][1]
            self.assertLessEqual(abs(s - base), 0.10 + 1e-9)


class TestShortlistGate(unittest.TestCase):
    def test_shortlist_k_restricts_the_candidate_set(self):
        store = CandidateStore()
        for i in range(10):
            store.add_candidate(Candidate(id=f"d{i}", content=f"doc {i}"))
        layer = ReputationLayer(store, weights=(0.70, 0.20, 0.10, 0.0), gamma=1.0)
        sims = {f"d{i}": 1.0 - 0.1 * i for i in range(10)}
        got = {r[0] for r in layer.rescore(sims, top_k=3, explore=False, shortlist_k=3).results}
        self.assertTrue(got <= {"d0", "d1", "d2"})


class TestGradedRewardAndNoise(unittest.TestCase):
    def test_graded_reward_is_not_silently_downweighted(self):
        """A verifier reporting 7/10 tests is a confident measurement, not a coin flip."""
        sig = OutcomeSignals(s_gt=0.7)
        self.assertAlmostEqual(calculate_kappa(0.7, sig, mode="decisiveness"), 0.4)
        self.assertAlmostEqual(calculate_kappa(0.7, sig, mode="auto"), 1.0)

    def test_partial_credit_moves_counters_more_than_binary_rounding(self):
        store = CandidateStore()
        store.add_candidate(Candidate(id="d", content="x", last_confirmed=0.0))
        _observe(store, "d", 0.7, t=1.0)
        self.assertGreater(store.get_candidate("d").alpha, 1.5)

    def test_noise_correction_softens_a_failure_from_a_lossy_verifier(self):
        noise = VerifierNoise(rho_fp=0.05, rho_fn=0.30)
        self.assertGreater(correct_outcome(0.0, noise), 0.0)
        self.assertLess(correct_outcome(1.0, noise), 1.0)
        self.assertEqual(correct_outcome(0.0, None), 0.0)

    def test_uninformative_verifier_yields_no_information(self):
        self.assertAlmostEqual(correct_outcome(0.0, VerifierNoise(0.5, 0.5)), 0.5)


class TestAttribution(unittest.TestCase):
    def test_generation_failure_barely_blames_the_document(self):
        store = CandidateStore()
        store.add_candidate(Candidate(id="r", content="x", last_confirmed=0.0))
        store.add_candidate(Candidate(id="g", content="x", last_confirmed=0.0))
        _observe(store, "r", 0.0, t=1.0, attribution=Attribution.RETRIEVAL)
        _observe(store, "g", 0.0, t=1.0, attribution=Attribution.GENERATION)
        self.assertGreater(
            store.get_candidate("r").beta - 1.0,
            10.0 * (store.get_candidate("g").beta - 1.0),
        )

    def test_verifier_failure_is_not_evidence_at_all(self):
        store = CandidateStore()
        store.add_candidate(Candidate(id="d", content="x", last_confirmed=0.0))
        _observe(store, "d", 0.0, t=1.0, attribution=Attribution.VERIFIER)
        c = store.get_candidate("d")
        self.assertEqual((c.alpha, c.beta), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
