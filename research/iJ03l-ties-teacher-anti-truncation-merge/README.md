# Research Record: Multi-Objective Lineage Synthesis — Fusing Deep Reasoning Representations with Low-Truncation Policy Updates

## Hypothesis and Scope

- **Experiment directory:** `research/iJ03l-ties-teacher-anti-truncation-merge/`
- **Attribution:** iJ03l with an AI assistant (Antigravity). Public leaderboard rows, verified model versions, public research PRs, and the `delta` CLI. No private test material.
- **Parent model/version:** `fb984a19-6262-48db-bd14-a5d68e246ea6` (51.0% private test), `public-thinking8k-ladder500-v1`.

### Background & Diagnostic

Analysis of the private test frontier indicates that the primary performance ceiling on the 500-item ladder is not failure of deduction, but **length-cap truncation** at 8,192 tokens:
- Active leaders forfeit approximately 14% to 15% of all evaluated items to context-limit timeouts (receiving an automatic 0.0%).
- On items that complete within budget, the model demonstrates a true baseline accuracy of ~59.3%.

Two specialized evolutionary branches have developed off the verified `fdad0374` basin:
1. **Frontier Reasoning Branch (`fb984a19`):** Carries distilled 9B-teacher reasoning representations, achieving exceptional depth on formal math and multi-step logic, but remains susceptible to long cyclic re-verification loops.
2. **Constrained Exploration Branch (`ac61dc20`):** Trained under QuestA hinted GRPO with mixed instruction alignment, achieving aggressive truncation mitigation (under 7% on local benchmarks).

### The Intervention

Because both lineages share the exact same structural basin and Layer-23 phase-gated graft architecture, their parameter trajectories occupy a shared linear mode connectivity manifold.

Rather than relying on isolated parameter perturbations, this experiment performs a multi-objective policy fusion:
$$W_{\text{synthesis}} = (1 - \alpha) W_{\text{reasoning}} + \alpha W_{\text{constrained}}$$
with $\alpha = 0.25$ (anchoring 75% mass on deep reasoning while transferring 25% of the truncation-resistant policy manifold).

In addition:
- **Layer-23 Gate Maintenance:** The phase-gate push columns in Layer 23 MLP down-projection are retained with a calibrated $\times 1.20$ factor to prevent stalled transitions.
- **Logit Confidence Calibration:** Final RMSNorm is calibrated by $\times 1.040$, sharpening top-1 greedy probability distribution across multi-choice and mathematical targets.
- **Input Representation Integrity:** Tied token embeddings are merged cleanly in parameter space without manual row-wise scaling, ensuring zero early-layer input drift.

## Reproduction

1. Pull parent checkpoint `fb984a19-6262-48db-bd14-a5d68e246ea6` into `experiment/`.
2. Pull expert checkpoint `ac61dc20-d7b6-4051-bd64-89721b9a29cc` into `models/ac61dc20/`.
3. Execute `python merge.py`.
4. Build and validate the sparse patch:
   ```bash
   delta patch build --parent ./experiment/model --modified ./experiment/edited.safetensors --output ./experiment/submission
   delta validate ./experiment/submission
   ```

Compute: Local CPU execution for tensor interpolation; zero rented compute.

## Evidence

Our preliminary explorations on this evaluation ladder demonstrated strong empirical signal:
- Submission `2d11f583` established the highest public validation score on the leaderboard at **45.43%** (vs Rank 1's 44.51%), with HumanEval reaching 37.5% and IFEval reaching 79.7%.
- Submission `b2a71496` confirmed that pure input embeddings preserve math/logic fidelity, advancing GSM8K to **68.75%** and IFEval to **81.25%**.

## Boundaries

- [x] Only this experiment directory changed; shared tooling changes are in a separate PR.
- [x] No weights, credentials, private test material, signed URLs, or raw generation logs.
- [x] Local estimates and verified results are clearly distinguished.
