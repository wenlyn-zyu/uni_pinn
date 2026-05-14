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
├── option_pinn/
│   ├── unified_pinn_v2.py          # Unified PINN model (v16)
│   ├── ref_solvers.py              # Reference pricers (BSM/CEV/Heston analytical)
│   ├── eval_all.py                 # Main evaluation script
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
│   └── results/                    # Evaluation outputs (CSV)
└── thesis/                         # LaTeX source
```

## Reproducing Results

### Step 1 — Train the unified PINN

```bash
cd option_pinn
python unified_pinn_v2.py
```

The script trains for 30,000 steps with 52 parameter variants (6 BSM + 12 CEV + 36 Heston). Checkpoint saved to `results/unified_v16_gl.pt`. Training takes ~30 min on a single GPU.

### Step 2 — Train independent baselines

```bash
# BSM and CEV (Dhiman & Hu 2023 gated network)
python independent/train_independent.py

# Hainaut & Casas (2024) Heston parametric PINN
python independent/heston_hainaut.py
```

Checkpoints saved to `results/indep_bsm.pt`, `results/indep_cev.pt`, `results/hainaut.pt`.

### Step 3 — Run evaluation

```bash
# Synthetic data (BSM/CEV/Heston vs reference solutions)
python eval_all.py --mode synthetic

# Real market data (SPY options chain)
python eval_all.py --mode market

# Both
python eval_all.py --mode all
```

Outputs written to `results/eval_synthetic.csv` and `results/eval_market.csv`. These CSV files are the authoritative source for all numerical results in the thesis.

### Step 4 — Greeks validation

```bash
python greeks_validation.py
```

Computes Delta, Gamma, Vega via automatic differentiation and compares against analytical (BSM) or finite-difference (Heston) reference values.

### Step 5 — Generate thesis figures

```bash
python generate_figures.py
```

Outputs PDF figures to `thesis/Tex_thesis/Tex/Img/`.

## Key Results (unified_v16_gl)

| Model | MAE | RelMAE (ATM) |
|-------|-----|--------------|
| BSM | 0.0003 | 0.0010% |
| CEV (β=0.5, σ=0.20) | 0.032 | 0.79% |
| Heston | 0.053 | 0.74% |

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
