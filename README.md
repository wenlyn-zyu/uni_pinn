# Unified Option Pricing Framework with PINN and LLM

A unified Physics-Informed Neural Network (PINN) that solves BSM, CEV, and Heston option pricing PDEs simultaneously, with a Large Language Model routing layer for natural language interaction.

## Requirements

```bash
conda activate pinn_option
```

Key dependencies: PyTorch, NumPy, SciPy, pandas, requests (for LLM API).

## Project Structure

```
uni_pinn/
├── README.md
├── option_pinn/
│   ├── unified_pinn_v2.py          # Unified PINN model (v16)
│   ├── ref_solvers.py              # Reference pricers (BSM/CEV/Heston analytical)
│   ├── eval_all.py                 # Synthetic and market evaluation
│   ├── llm_test.py                 # LLM routing layer test (20 cases)
│   ├── ablation.py                 # Ablation study (3 variants)
│   ├── greeks_validation.py        # Greeks computation and validation
│   ├── generate_figures.py         # Thesis figures
│   ├── llm_router.py               # LLM routing layer (DeepSeek API)
│   ├── independent/                # Independent PINN baselines
│   │   ├── bsm_pinn.py             # BSM gated network (Dhiman & Hu 2023)
│   │   ├── cev_pinn.py             # CEV gated network
│   │   ├── heston_hainaut.py       # Hainaut & Casas (2024) parametric PINN
│   │   └── train_independent.py    # Train BSM/CEV baselines
│   ├── data/
│   │   └── spy_quotedata.csv       # SPY options chain (2026-05-11, S=739.75)
│   └── results/                    # Evaluation outputs (CSV, model weights .pt)
└── thesis/                         # LaTeX source
```

## Reproducing Results

### Step 1 — Train the unified PINN

```bash
cd option_pinn
python unified_pinn_v2.py
```

The script trains for 30,000 steps with 54 parameter variants (6 BSM + 12 CEV + 36 Heston). Checkpoint saved to `results/unified_v16_gl.pt`. Training takes ~30 min on a single GPU.

### Step 2 — Train independent baselines and parametric PINN

```bash
# BSM and CEV (Dhiman & Hu 2023 gated network)
python independent/train_independent.py

# Hainaut & Casas (2024) Heston parametric PINN
python independent/heston_hainaut.py

# Fully parametric PINN (all three models, wide parameter range)
python parametric_pinn/train_parametric.py
```

Checkpoints saved to `results/indep_bsm.pt`, `results/indep_cev.pt`, `results/hainaut.pt`, and `parametric_pinn/results/fully_param_v1.pt`.

### Step 3 — Run evaluation

```bash
# Synthetic data (BSM/CEV/Heston vs reference solutions)
python eval_all.py --mode synthetic

# Real market data (SPY options chain)
python eval_all.py --mode market

# Both
python eval_all.py --mode all
```

Outputs written to `results/eval_synthetic.csv` and `results/eval_market.csv`.

### Step 4 — LLM routing layer test (20 test cases)

```bash
export DEEPSEEK_API_KEY=your_key_here   # or set in llm_router.py
python llm_test.py --save results/llm_test_results.json
```

Runs 20 test cases (10 Chinese + 10 English, covering BSM/CEV/Heston) and reports model selection accuracy. Expected result: 20/20 (100%).

### Step 5 — Ablation study

```bash
python ablation.py --epochs 30000 --save results/ablation_results.json
```

Trains three ablation variants (no soft mask / no additive parameterization / no data anchor) and reports BSM and Heston MAE for each. Takes ~7 hours on a single GPU (3 × 30,000 steps). Use `--epochs 5000` for a quick sanity check.

### Step 6 — Greeks validation

```bash
python greeks_validation.py
```

### Step 7 — Generate thesis figures

```bash
python generate_figures.py
```

Outputs PDF figures to `thesis/Tex_thesis/Tex/Img/`.

## Key Results (unified_v16_gl)

| Model | MAE | RelMAE (ATM) |
|-------|-----|--------------|
| BSM | 0.0005 | 0.0009% |
| CEV (β=0.5, σ=0.20) | 0.032 | 0.79% |
| Heston | 0.102 | 0.599% |

Inference: ~2 ms (GPU), ~8 ms (CPU). LLM routing: 100% accuracy on 20 test cases.

## LLM Routing Layer

Requires a DeepSeek API key set as environment variable:

```bash
export DEEPSEEK_API_KEY=your_key_here
python llm_router.py
```

Accepts natural language input (Chinese or English) and returns structured JSON with model type and parameters, which is then passed to the unified PINN for pricing.

## Model Parameters (thesis experiments)

| Parameter | BSM | CEV | Heston |
|-----------|-----|-----|--------|
| K | 100 | 100 | 100 |
| T | 1 yr | 1 yr | 1 yr |
| r | 0.05 | 0.05 | 0.05 |
| σ | 0.20 | 0.20 | — |
| β | 1.0 | 0.5 | — |
| κ | — | — | 2.0 |
| θ | — | — | 0.04 |
| ξ | — | — | 0.3 |
| ρ | — | — | −0.7 |
| v₀ | — | — | 0.04 |

## Reference

Thesis: *A Unified Option Pricing Framework Based on Physics-Informed Neural Networks and Large Language Models*, Wenlong Zhu, ShanghaiTech University, 2026.
