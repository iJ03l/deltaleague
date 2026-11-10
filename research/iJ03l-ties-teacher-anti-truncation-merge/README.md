# Research Record: Middle-Out Spectral Wiener Compression, Output-Only Answer Subspace Projection, and Phase-Gate Calibration

## Hypothesis and Scope

- **Experiment directory:** `research/iJ03l-ties-teacher-anti-truncation-merge/`
- **Attribution:** iJ03l with an AI assistant (Antigravity). Public leaderboard rows, verified model versions, public research PRs, and the `delta` CLI. No private test material.
- **Parent model/version:** `fb984a19-6262-48db-bd14-a5d68e246ea6` (Rank 1, 51.0% private test), `public-thinking8k-ladder500-v1`.

### 1. Structural & Spectral Diagnostics

Layer-wise singular value scanning across all 24 blocks of Qwen3.5-0.8B (`0ea62f40` and `fb984a19`) revealed three empirical mechanisms governing reasoning length and completion:

1. **Late-Middle High-Rank Variance Bottleneck (`Layers 16–22`):**
   - `mlp.down_proj` StableRank jumps from `83.4–101.6` in Layers `0–10` to **`177.0–210.3`** in Layers `16–22` (peaking at `210.26` in Layer 21).
   - `attn_out` (`linear_attn.out_proj` / `self_attn.o_proj`) StableRank simultaneously climbs from `43.5–90.8` in `0–13` to **`108.5–139.4`** in Layers `19–22` (peaking at `139.44` in Layer 21).
   - Over `98.5%` of matrix Frobenius energy in `Layers 18–22` resides above the flat spectral bulk tail, which sustains low-confidence re-verification loops.
2. **Decoupled Answer Subspace (`6.78×` Selectivity):**
   - SVD on the 17 primary target answer tokens (`0–9`, `A–E`, `True`/`False`) in `embed_tokens.weight` shows their Top-5 singular subspace captures **`78.62%`** of answer-token energy vs **`11.59%`** across random vocabulary (`6.78×` selectivity), with individual residual coordinates reaching `5.27×` selectivity.
   - Modifying tied `embed_tokens` directly boosts short-context validation (`45.43%` in submission `2d11f583`) but perturbs the unnormalized Layer-0 input residual stream on arithmetic tokens. Projecting that selectivity instead onto `model.language_model.norm.weight` restricts the sharpening strictly to the output LM head.
3. **Layer-23 Finite-Difference Phase Gate (`Neurons 1998 & 3529`):**
   - Exhaustive scanning of Layer 23 MLP (`3,584` neurons) locates the `</think>` SwiGLU phase-gate pair at indices **`1998` and `3529`** (`gate_proj` row norms `13.217` and `13.211`, an `8.8×` outlier; `up_proj` cosine `+1.0000`; `down_proj` cosine `-1.0000`, `cos(</think>) = ±0.4558`). Scaling both `down_proj` columns symmetrically preserves exact out-of-thinking cancellation while amplifying `</think>` termination impulse when triggered.

### 2. The Intervention (`Candidate I`)

1. **Middle-Out Smooth Spectral Wiener Compression (`Layers 10–22`, centered at `L=20`, $\sigma=2.5$):**
   Applied to both `mlp.down_proj` and `attn_out` (`out_proj` / `o_proj`) with Frobenius energy preservation (`--preserve-norm`), $\lambda_{\text{Wiener}} = 0.28$, filter power $p = 4$:
   $$w_\ell = \exp\!\left(-\frac{(\ell - 20)^2}{2 \cdot 2.5^2}\right), \qquad \tilde{\sigma}_i = \sigma_i \cdot \frac{\sigma_i^4}{\sigma_i^4 + (\lambda w_\ell \sigma_{\text{med}})^4}$$
   This gently damps flat high-rank tail variance in `Layers 18–22` (`98.54%–99.99%` energy kept per block) while leaving early layers (`0–9`) and Layer `23` untouched.
2. **Output-Only Answer Coordinate Selectivity:**
   Zero-mean log-energy selectivity across `[1024]` applied to `model.language_model.norm.weight` (`ans_eps = 0.045`, global `norm_gain = 1.022`), keeping `embed_tokens.weight` untouched.
3. **Symmetric Layer-23 Graft Tuning:**
   `layers.23.mlp.down_proj[:, [1998, 3529]] *= 1.25`.

## Reproduction

From an initialized workspace containing `fb984a19` in `./experiment/model`:

```bash
python middle_out_compress.py \
  --center 20.0 --sigma 2.5 --lambda-wiener 0.28 --preserve-norm --compress-attn \
  --norm-gain 1.022 --ans-eps 0.045 --l23-kappa 1.25 \
  --output ./experiment/edited_mo_i.safetensors

delta patch build --parent ./experiment/model --modified ./experiment/edited_mo_i.safetensors --output ./experiment/submission --force
delta validate ./experiment/submission
```

Compute: Local CPU SVD and inference diagnostics; zero rented compute.

## Evidence

### Local Advisory Paired Benchmark (`enable_thinking=True`, `presence_penalty=1.5`, `max_new_tokens=1024`, greedy, 6 items)

| Variant | Score | Closed `</think>` | Avg Think Tok | Avg Total Tok | Notes |
|---|---:|---:|---:|---:|---|
| `fb984a19` Baseline (Rank 1) | `3/6` | `4/6` | `558.0` | `757.3` | `bbh_logic` timed out (`>1024`), `ifeval` rambled (`1024`) |
| `0c834b41` Reference (`11.4%` trunc) | `3/6` | `6/6` | `496.0` | `714.7` | `bbh_logic` ok (`614`), `ifeval` rambled (`1024`) |
| `026e45ad` Repro (`45.43%` Val) | `3/6` | `5/6` | `471.8` | `680.0` | `ifeval` ok (`690`), `bbh_logic` timed out |
| **Candidate I (`middle_out_compress.py`)** | **`4/6`** | **`5/6`** | **`469.2`** | **`559.7`** | **`bbh_logic` ok (`521`), `ifeval` ok (`421`), `-26.1%` total tokens** |

### Prior Verified Leaderboard Milestones
- **Submission `2d11f583` (`026e45ad`):** Public validation **45.43%** (HumanEval `37.5%`, IFEval `79.7%`, BBH `42.6%`, `13.2%` truncation), private test **49.0%**.
- **Submission `9030885e` (`7c2363b2`):** Multi-objective synthesis reduced truncation to `13.6%`, private test **49.20%**.
- **Submission `28739667` (`ed536971`):** Task-vector SVD rank compression ($r=16, \beta=0.40$) reached **50.20%** private test (**Rank 4**) and **44.56%** validation (HumanEval `40.63%`, BBH `38.89%`).

## Boundaries

- [x] Only this experiment directory changed; shared tooling changes are in a separate PR.
- [x] No weights, credentials, private test material, signed URLs, or raw generation logs.
- [x] Local estimates and verified results are clearly distinguished.
