import os
import random
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sim")))

import numpy as np  # noqa: E402

import run_gate_matrix as gm  # noqa: E402
from rrl.feedback import OutcomeSignals  # noqa: E402

POOL = [(1, 0.9), (2, 0.6), (3, 0.3)]
ORDER = {0: [1, 2, 3]}


def run(name, outcome, steps=60, coverage=1.0, self_report=None, seed=0, pool=POOL):
    """Replays `name` on one request; returns the tools it chose. A lying self-report leaves
    the judge honest."""
    order = {0: [d for d, _ in sorted(pool, key=lambda p: -p[1])]}
    choose, update = gm.make_policy(name, seed, order, {0: pool})
    rng = random.Random(seed)
    picks = []
    for step in range(steps):
        d = choose(0, pool, step)
        y = outcome(d)
        report = y if self_report is None else self_report(d, y)
        update(
            0,
            d,
            OutcomeSignals(s_gt=y if rng.random() < coverage else None, s_behave=report, s_judge=y),
            step,
        )
        picks.append(d)
    return picks


class TestPerContextMemory(unittest.TestCase):
    def test_tool_keyed_memory_matches_position_semantics(self):
        # Tool 1 fails, tool 2 works: walk the reranker order once, then stay on tool 2.
        picks = run("query_memory", lambda d: 1.0 if d == 2 else 0.0, steps=5)
        self.assertEqual(picks, [1, 2, 2, 2, 2])
        # A failure moves to the next tool in order, wrapping around.
        picks = run("query_memory", lambda d: 0.0, steps=4)
        self.assertEqual(picks, [1, 2, 3, 1])

    def test_memory_follows_the_tool_when_the_order_changes(self):
        choose, update = gm.make_policy("query_memory", 0, {0: [1, 2, 3]})
        self.assertEqual(choose(0, POOL, 0), 1)
        update(0, 1, OutcomeSignals(s_gt=0.0), 0)
        self.assertEqual(choose(0, POOL, 1), 2)
        update(0, 2, OutcomeSignals(s_gt=1.0), 1)
        # A new tool ranked first does not displace the tool that worked.
        order = {0: [9, 2, 1, 3]}
        choose, update = gm.make_policy("query_memory", 0, order)
        update(0, 9, OutcomeSignals(s_gt=0.0), 0)
        self.assertEqual(choose(0, [(9, 0.95)] + POOL, 1), 2)


class TestMemRLOfficial(unittest.TestCase):
    def test_learns_from_plus_minus_one_rewards(self):
        picks = run("memrl_official", lambda d: 1.0 if d == 3 else 0.0, steps=80)
        self.assertGreater(picks[-40:].count(3), 30)

    def test_drops_candidates_below_q_min(self):
        choose, update = gm.make_policy("memrl_official", 0, dict(ORDER), {0: POOL})
        for step in range(6):  # Q: -0.3, -0.51, -0.66, -0.76, -0.83 -> below -0.8
            update(0, 1, OutcomeSignals(s_gt=0.0), step)
        picks = {choose(0, POOL, s) for s in range(200)}
        self.assertNotIn(1, picks)

    def test_takes_unverified_feedback_at_face_value(self):
        # Tool 1 never works but it and the judge report success; unverified, it stays.
        choose, update = gm.make_policy("memrl_official", 0, dict(ORDER), {0: POOL})
        picks = []
        for step in range(60):
            d = choose(0, POOL, step)
            update(0, d, OutcomeSignals(s_behave=1.0, s_judge=1.0), step)
            picks.append(d)
        self.assertGreater(picks.count(1), 45)


CLOSE = [(1, 0.9), (2, 0.85), (3, 0.8)]


class TestRATED(unittest.TestCase):
    def test_learns_tool_reliability_from_execution(self):
        picks = run("rated", lambda d: 1.0 if d == 3 else 0.0, steps=150, pool=CLOSE)
        self.assertGreater(picks[-50:].count(3), 45)

    def test_semantic_score_dominates_far_candidates(self):
        # 0.75 * semantic outweighs the 0.15 bandit term: a working tool far below the top is
        # never reached, as in RATED, whose top-K candidates are semantically close.
        picks = run("rated", lambda d: 1.0 if d == 3 else 0.0, steps=100)
        self.assertEqual(picks.count(3), 0)

    def test_verified_variant_is_not_fooled_by_unverified_success(self):
        # Tool 1 fails but always reports success (the judge too); 20% of calls are verified.
        def lying(d, y):
            return 1.0 if d == 1 else y

        def outcome(d):
            return 1.0 if d == 2 else 0.0

        kw = dict(steps=300, coverage=0.2, pool=CLOSE)
        plain, verified = [], []
        for seed in range(5):
            plain += run("rated", outcome, self_report=lying, seed=seed, **kw)[-100:]
            verified += run("rated_verified", outcome, self_report=lying, seed=seed, **kw)[-100:]
        self.assertGreater(verified.count(2), plain.count(2) + 100)


class TestRawCosines(unittest.TestCase):
    def tearDown(self):
        gm.set_semantic(None)

    def test_rated_uses_raw_cosines_when_given(self):
        # Min-max normalised pool scores exaggerate the gap; on raw cosines RATED can learn.
        pool = [(1, 1.0), (2, 0.0)]

        def outcome(d):
            return 1.0 if d == 2 else 0.0

        normalised = run("rated", outcome, steps=100, pool=pool)
        gm.set_semantic({0: {1: 0.62, 2: 0.60}})
        raw = run("rated", outcome, steps=100, pool=pool)
        self.assertEqual(normalised.count(2), 0)
        self.assertGreater(raw[-50:].count(2), 45)


class TestCESA(unittest.TestCase):
    def test_distrusts_a_source_that_contradicts_audits(self):
        # The self-report always says success; the judge is honest; the audit is on every call.
        picks = run(
            "cesa",
            lambda d: 1.0 if d == 3 else 0.0,
            steps=400,
            self_report=lambda d, y: 1.0,
        )
        self.assertGreater(picks[-40:].count(3), 35)

    def test_learns_quickly_when_sources_are_honest(self):
        picks = run("cesa", lambda d: 1.0 if d == 3 else 0.0, steps=200)
        self.assertGreater(picks.count(3), 180)


class TestFeatureVectors(unittest.TestCase):
    def test_context_vectors_are_unit_and_fixed(self):
        v = gm.context_vector(("site", 3))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=9)
        self.assertTrue(np.array_equal(v, gm.context_vector(("site", 3))))
        self.assertFalse(np.array_equal(v, gm.context_vector(("site", 4))))


if __name__ == "__main__":
    unittest.main()
