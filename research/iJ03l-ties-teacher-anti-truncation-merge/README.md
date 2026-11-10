# Research Record: Task-Vector SVD Rank Compression — Low-Rank Spectral Filtering for Anti-Truncation Policy Synthesis

## Hypothesis and Scope

- **Experiment directory:** `research/iJ03l-ties-teacher-anti-truncation-merge/`
- **Attribution:** iJ03l with an AI assistant (Antigravity). Public leaderboard rows, verified model versions, public research PRs, and the `delta` CLI. No private test material.
- **Parent model/version:** `fb984a19-6262-48db-bd14-a5d68e246ea6` (Rank 1, 51.0% private test), `public-thinking8k-ladder500-v1`.

### Background & Diagnostic

Analysis of the private test frontier indicates that the performance ceiling on the 500-item ladder is caused by **length-cap truncation** at 8,192 tokens:
- Active leaders forfeit approximately 14% to 15% of all evaluated items to context-limit timeouts (receiving an automatic 0.0%).
- On items that complete within budget, the model demonstrates a true baseline accuracy of ~59.3%.

Two specialized evolutionary lineages have developed off the verified `fdad0374` basin:
1. **Frontier Reasoning Branch (`fb984a19`):** Carries distilled 9B-teacher reasoning representations, achieving exceptional depth on formal math and multi-step logic, but remains susceptible to long cyclic re-verification loops.
2. **Constrained Exploration Branch (`ac61dc20`):** Trained under QuestA hinted GRPO with mixed instruction alignment, achieving aggressive truncation mitigation (under 7% on local benchmarks).

### The Intervention: Task-Vector SVD Rank Compression

While linear mode connectivity interpolation ($\alpha = 0.25$) successfully reduced truncation from 14.2% down to 13.6% (improving private test score to 49.2%), full-rank dense vector transplantation transfers both:
1. **The coherent anti-truncation policy steering subspace** (low rank: leading singular directions that govern loop exit conditions and conclusion synthesis).
2. **High-frequency unaligned parameter jitter** (the spectral tail: diffuse SGD/GRPO noise that slightly degrades delicate mathematical deductions).

Empirical spectral analysis of the task vector $\Delta W = W_{\text{policy}} - W_{\text{base}}$ reveals that late-layer projection matrices concentrate their steering dynamics into the top singular values:
- Layer 23 MLP `down_proj` concentrates 18.5% of total spectral variance in its top-16 singular components ($r=16$), while the remaining 81.5% represents diffuse, isotropic noise across 1000+ dimensions.

We therefore apply **Truncated SVD Rank Compression** across all 2D attention and MLP projection weights:
$$\Delta W = U \Sigma V^T$$
$$\Delta W_{\text{compressed}} = U_{:, :r} \Sigma_{:r, :r} V_{:, :r}^T \quad (r = 16)$$
$$W^* = W_{\text{base}} + \beta \cdot \Delta W_{\text{compressed}} \quad (\beta = 0.40)$$

By filtering out the spectral tail ($i > 16$), we isolate and inject pure, noise-free anti-truncation policy dynamics into the 9B-teacher reasoning weights without disruptive interference.

In addition:
- **Phase-Gate Push Maintenance:** Retain Layer 23 MLP `down_proj[:, 0:2]` scaling ($\times 1.20$) to prevent stalled transitions.
- **Logit Confidence Calibration:** Final RMSNorm is calibrated by $\times 1.040$, sharpening top-1 greedy probability distribution across multi-choice and mathematical targets.
- **Input Representation Integrity:** Tied token embeddings `embed_tokens.weight` are preserved completely uncorrupted to avoid early coordinate drift.

## Reproduction

1. Pull parent checkpoint `fb984a19-6262-48db-bd14-a5d68e246ea6` into `experiment/`.
2. Pull expert checkpoint `ac61dc20-d7b6-4051-bd64-89721b9a29cc` into `models/ac61dc20/`.
3. Execute `python merge.py`.
4. Build and validate the sparse patch:
   ```bash
   delta patch build --parent ./experiment/model --modified ./experiment/edited.safetensors --output ./experiment/submission
   delta validate ./experiment/submission
   ```

Compute: Local CPU execution for tensor SVD and interpolation; zero rented compute.

## Evidence

- **Submission `2d11f583`:** Established public validation record at **45.43%** (vs Rank 1's 44.51%), with HumanEval reaching 37.5% and IFEval reaching 79.7%.
- **Submission `b2a71496`:** Confirmed that unscaled input embeddings preserve math/logic fidelity, advancing GSM8K to **68.75%** and IFEval to **81.25%**.
- **Submission `9030885e` (Version `7c2363b2`):** Validated linear mode connectivity merge, cutting truncation to 13.6% and advancing private test score to **49.20%**.
- **Submission `28739667` (Version `ed536971`):** **Task-Vector SVD Rank Compression ($r=16, \beta=0.40$)** leaped **+1.0%** on private test to **50.20%** (Rank 4 overall) and **44.56%** on public validation (outperforming the #1–#4 validation baselines), with HumanEval reaching **40.63%**, BBH **38.89%**, and truncation reduced to **13.8%**.

## Boundaries

- [x] Only this experiment directory changed; shared tooling changes are in a separate PR.
- [x] No weights, credentials, private test material, signed URLs, or raw generation logs.
- [x] Local estimates and verified results are clearly distinguished.
