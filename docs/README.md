# Titanium Technical Documentation & Research

Engineering specifications, empirical ablation studies, NNUE/tuple architecture benchmarks, search tuning reports, competitive analyses, and reproducible runbooks for the **Titanium** Ataxx engine.

---

## 1. Core Specifications & Protocols
- **[Experiment Ledger (LEDGER.md)](../LEDGER.md)** — Canonical Fishtest-style log of all gated neural networks, search changes, and competitive benchmarks.
- **[Gate Protocol (GATE_RULE.md)](./GATE_RULE.md)** — SPRT promotion rules, paired opening book methodology, and `ti insight` / `ti analyze` verification standards.
- **[Dataset Guide (DATASET_GUIDE.md)](./DATASET_GUIDE.md)** — Binary dataset formats (`.tgam`, `.tbst`, `.tdat`), sequence DAG deduplication, and labeling standards.
- **[Gate Registry (GATE_RUNS.tsv)](./GATE_RUNS.tsv)** | **[TSTG Models (TSTG_MODELS.tsv)](./TSTG_MODELS.tsv)** | **[Pattern Models (PATTERN_MODELS.tsv)](./PATTERN_MODELS.tsv)** — Machine-generated run and artifact registries.
- **[Disagreement Head Spec (TNN3 + TSN1)](./DISAGREEMENT_HEAD_SPEC.md)** — Dual-headed NNUE valuation + volatility prediction.
- **[Persistent Search Tree vs. One-Shot Botzone](./PERSISTENT_VS_ONESHOT_BOTZONE.md)** — Across-turn TT aging, fail-low preservation, and Botzone timing analysis.

---

## 2. Research Reports (`research/`)

| Area | Directory | Highlights |
|---|---|---|
| **NNUE & Architecture** | [`research/nnue-eval/`](./research/nnue-eval/) | [RESEARCH_ARCH_HUNT.md](./research/nnue-eval/RESEARCH_ARCH_HUNT.md), [RESEARCH_KNOWLEDGE_PER_INFERENCE.md](./research/nnue-eval/RESEARCH_KNOWLEDGE_PER_INFERENCE.md), [RESEARCH_SMALLNET_ARCH.md](./research/nnue-eval/RESEARCH_SMALLNET_ARCH.md), [NET_ANALYSIS.md](./research/nnue-eval/NET_ANALYSIS.md), [NNUE_EFFICIENCY_PLAN.md](./research/nnue-eval/NNUE_EFFICIENCY_PLAN.md), [RESEARCH_NNUE_V10_2026_11_15.md](./research/nnue-eval/RESEARCH_NNUE_V10_2026_11_15.md), [GOLDEN_DOSE.md](./research/nnue-eval/GOLDEN_DOSE.md), [EVAL_FAILURES.md](./research/nnue-eval/EVAL_FAILURES.md), [FUTILITY_DISAGREEMENT_EXPERIMENT.md](./research/nnue-eval/FUTILITY_DISAGREEMENT_EXPERIMENT.md) |
| **Tuples, HCE & Speed** | [`research/hce-patterns-speed/`](./research/hce-patterns-speed/) | [RESEARCH_JATAXX.md](./research/hce-patterns-speed/RESEARCH_JATAXX.md), [RESEARCH_TUPLES_HCE.md](./research/hce-patterns-speed/RESEARCH_TUPLES_HCE.md), [TWO_STAGE_TUPLE.md](./research/hce-patterns-speed/TWO_STAGE_TUPLE.md), [RESEARCH_ULTIMATE_HCE.md](./research/hce-patterns-speed/RESEARCH_ULTIMATE_HCE.md), [RESEARCH_HCE_FIT.md](./research/hce-patterns-speed/RESEARCH_HCE_FIT.md), [SEE_EVAL_MEASUREMENT.md](./research/hce-patterns-speed/SEE_EVAL_MEASUREMENT.md), [TSTG_EXPERIMENT.md](./research/hce-patterns-speed/TSTG_EXPERIMENT.md) |
| **Search & Profiling** | [`research/search/`](./research/search/) | [ENGINE_PROFILE_2026_11_17.md](./research/search/ENGINE_PROFILE_2026_11_17.md), [ENGINE_REVIEW_2026_11_18.md](./research/search/ENGINE_REVIEW_2026_11_18.md), [RESEARCH_FUNDAMENTALS_REVIEW.md](./research/search/RESEARCH_FUNDAMENTALS_REVIEW.md), [RESEARCH_SEARCH_LITERATURE.md](./research/search/RESEARCH_SEARCH_LITERATURE.md), [RESEARCH_TREE_SHAPE.md](./research/search/RESEARCH_TREE_SHAPE.md), [RESEARCH_LMR2.md](./research/search/RESEARCH_LMR2.md), [SPSA_STAGE3_REPORT.md](./research/search/SPSA_STAGE3_REPORT.md), [UBFM_PLAN.md](./research/search/UBFM_PLAN.md) |
| **Policy Head** | [`research/policy/`](./research/policy/) | [RESEARCH_POLICY_HEAD.md](./research/policy/RESEARCH_POLICY_HEAD.md), [RESEARCH_ORDERING_ORACLE.md](./research/policy/RESEARCH_ORDERING_ORACLE.md), [TEACHER_POLICY.md](./research/policy/TEACHER_POLICY.md), [POLICY_PLAN.md](./research/policy/POLICY_PLAN.md) |
| **Data & Training** | [`research/data-training/`](./research/data-training/) | [DATASET_V10.md](./research/data-training/DATASET_V10.md), [KATA_FIRST_V3.md](./research/data-training/KATA_FIRST_V3.md), [ML_TRAINING_RESEARCH.md](./research/data-training/ML_TRAINING_RESEARCH.md), [TRAINING_METHODS.md](./research/data-training/TRAINING_METHODS.md), [STOCKFISH_LC0_RECIPE.md](./research/data-training/STOCKFISH_LC0_RECIPE.md), [PIPELINE_AUDIT.md](./research/data-training/PIPELINE_AUDIT.md) |
| **Proof hunting** | [`research/proof-hunt/`](./research/proof-hunt/) | [runbooks/PROOF_HUNT.md](./runbooks/PROOF_HUNT.md) (START HERE: tools, commands, checks), [ZATAXX_BZ_MODIFICATION.md](./research/proof-hunt/ZATAXX_BZ_MODIFICATION.md) |
| **Botzone & Opponents** | [`research/opponents-botzone/`](./research/opponents-botzone/) | [ENGINES.md](./research/opponents-botzone/ENGINES.md), [BOTZONE_GAME_ANALYSIS.md](./research/opponents-botzone/BOTZONE_GAME_ANALYSIS.md), [ATAXXCPP_RESEARCH.md](./research/opponents-botzone/ATAXXCPP_RESEARCH.md), [KATA_ARENA.md](./research/opponents-botzone/KATA_ARENA.md), [COMMUNITY_RESEARCH_2026.md](./research/opponents-botzone/COMMUNITY_RESEARCH_2026.md) |
| **Strategy & Roadmaps** | [`research/strategy/`](./research/strategy/) | [STRENGTH_LEVERS_2026_11.md](./research/strategy/STRENGTH_LEVERS_2026_11.md), [RESEARCH_ROUTE.md](./research/strategy/RESEARCH_ROUTE.md), [RESEARCH_WHERE_ELO_2026_11_02.md](./research/strategy/RESEARCH_WHERE_ELO_2026_11_02.md), [WHAT_MATTERS.md](./research/strategy/WHAT_MATTERS.md) |

---

## 3. Runbooks, Scripts & Supporting Materials
- **[runbooks/PROOF_HUNT.md](./runbooks/PROOF_HUNT.md)**: proof hunting with the Botzone-rules Zataxx (`ti proof-lines`), from games to dataset rows.
- **[`runbooks/`](./runbooks/)**: Training, cloud relabeling, Modal GPU setup, Docker CPU sandboxing, and build/checklists.
- **[`analysis/`](./analysis/)**: Evaluation harnesses, benchmark scripts, gate queues, and analysis tools used to produce the numbers in the reports.
- **[`external/`](./external/)**: Third-party architecture references studied during development.
- **[`archive/`](./archive/)**: Historical design reviews and milestone plans ([INDEX.md](./archive/INDEX.md)).
