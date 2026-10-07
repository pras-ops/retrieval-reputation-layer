# Realistic Recurring Benchmark Results (MBPP)

This document presents the empirical validation results of the **Retrieval Reputation Layer (RRL)** against a strong static baseline on a realistic, recurring-query setting using the Google MBPP dataset.

## Experimental Setup

- **Dataset:** 974 programming tasks from the Google MBPP dataset, grouped into naturally recurring problem families (e.g., list manipulation, math functions, string processing, tuples, etc.).
- **Retriever Pool:** Hybrid retrieval combining sparse (BM25) and dense embeddings, generating a top-5 candidate list.
- **Static Baseline:** A production-grade Cross-Encoder reranker selecting the top candidate from the top-5 hybrid retrieval list (no online learning or outcome awareness).
- **RRL Configuration:** Hybrid retriever wrapped with RRL using **global Beta counters** and a **Thompson-sampling exploration policy**. Downstream correctness (`passed = 1.0` or `0.0`) observed via isolated unit-test execution.
- **Sweep Parameters:** 10 independent random seeds, 8 recurring epochs (30 steps per epoch, totaling 240 steps per seed).
- **LLM Generator:** Gemini API (`gemini-2.5-pro` / Vertex AI) producing python code solutions.

---

## Performance Summary

| Metric | Static (Cross-Encoder) | RRL (Ours) | Absolute Lift | Relative Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Pass Rate** | 0.540 [0.406, 0.674] | **0.569 [0.509, 0.628]** | **+2.9 pts** | **+5.4%** |
| **Late-Stage Pass Rate** | 0.542 [0.399, 0.685] | **0.590 [0.525, 0.654]** | **+4.8 pts** | **+8.9%** |

*Note: In brackets `[...]` are the 95% confidence intervals across the 10 seeds.*

---

## Detailed Performance Analysis

### 1. The Learning Curve & Late-Stage Separation
During the early epochs, RRL pays a small **exploration tax** to probe alternative retrieval candidates and gather outcome statistics. 
* **Early stage:** The accuracies start closer as RRL populates the Beta counters.
* **Late stage (epochs 5-8):** As the reputation counters converge, RRL shifts toward exploitation and the gap widens to +4.8 pts: **0.590 [0.525, 0.654]** for RRL versus **0.542 [0.399, 0.685]** for the static reranker.
* **Statistical significance — not yet established.** With 10 seeds, the 95% confidence intervals overlap for both the overall and the late-stage pass rate. This benchmark therefore shows a consistent *direction*, not a statistically established lift. RRL's results do vary less across seeds (CI width `0.119` vs `0.268` overall), but a narrower spread is not a significance test.
* **What would settle it:** more seeds (Gate B needed 30 before its intervals separated), or a paired per-seed test, which fits here because both arms run on the same seeds and problem sets.

### 2. Mitigation of Distractor Noise
The core failure mode of the static cross-encoder is its susceptibility to highly semantically similar but logically incorrect distractor chunks (e.g., a function with the same name but slightly different parameters).
* RRL successfully identifies these distractors when they fail downstream unit tests.
* RRL decays their reputation, ensuring they are penalised and evicted from top rankings in subsequent epochs.

### 3. Verification of Stated Boundaries
These results are consistent with the central conditional claim of RRL:
1. **Recurrence + Verifier = Win:** In recurring tasks (MBPP problem families) with a reliable verifier (unit tests), online reputation tracking points the same way as the controlled gates: ahead of a strong static reranker overall, and further ahead once the counters converge. At 10 seeds that lift is directional, not yet statistically significant.
2. **Exploration Trade-Off:** The exploration tax is minor, meaning that RRL is a viable candidate for production integration in environments where queries recur.

---

## Visualization
The comparison plots showing the cumulative pass rates and learning trajectories across the epochs have been saved to:
`sim/gate_recurring_comparison.png`
