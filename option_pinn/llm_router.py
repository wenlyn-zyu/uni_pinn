# option_pinn/llm_router.py
"""LLM路由层：通过DeepSeek API将自然语言转换为期权定价参数，调用统一PINN求解。

用法:
  python llm_router.py "定价一份欧式看涨期权，特斯拉现价250，行权价260，半年到期"
  python llm_router.py "price a European put on AAPL, S=220, K=210, T=0.5yr, sigma=0.25"
  python llm_router.py "AAPL230616C00150000"                                      # OCC代码

依赖: openai (联网), torch + unified_pinn_v2 (定价时需要)
"""

import os
import re
import sys
import json
import argparse
from datetime import date

from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = "sk-139b45a0f07c400a9b87f629932b0fb7"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# OCC option symbol decoding
# ---------------------------------------------------------------------------
# OCC format: {SYMBOL:≤6}{YY}{MM}{DD}{C/P}{STRIKE×1000:8digits}
# Example: AAPL230616C00150000 → AAPL call expiring 2023-06-16 strike $150.000

_OCC_RE = re.compile(
    r"^([A-Za-z]{1,6})"          # ticker (1-6 letters)
    r"(\d{2})(\d{2})(\d{2})"     # YY MM DD
    r"([CP])"                     # Call / Put
    r"(\d{8})$"                   # strike × 1000, zero-padded
)

def is_occ_symbol(text: str) -> bool:
    """Check whether *text* looks like a standalone OCC option symbol."""
    return bool(_OCC_RE.match(text.strip()))


def decode_occ_symbol(occ: str) -> dict:
    """Decode an OCC option symbol into its components.

    Returns dict with keys: ticker, expiry (YYYY-MM-DD), option_type (call/put),
    strike, T (years from today).  T is approximate and should be verified.
    """
    m = _OCC_RE.match(occ.strip().upper())
    if not m:
        raise ValueError(f"Invalid OCC option symbol: {occ}")

    ticker, yy, mm, dd, cp, strike_raw = m.groups()
    year  = 2000 + int(yy)
    month = int(mm)
    day   = int(dd)
    strike = int(strike_raw) / 1000.0
    option_type = "call" if cp == "C" else "put"

    expiry_date = date(year, month, day)
    today = date.today()
    T = max((expiry_date - today).days / 365.0, 0.001)

    return {
        "ticker":      ticker,
        "expiry":      expiry_date.isoformat(),
        "option_type": option_type,
        "strike":      strike,
        "T_approx":    round(T, 4),
    }


def _format_occ_for_llm(occ: str) -> str:
    """Decode OCC symbol and return a prompt fragment for the LLM."""
    d = decode_occ_symbol(occ)
    return (
        f"OCC option code: {occ}\n"
        f"Decoded fields (LOCKED — output these exact values in JSON):\n"
        f'  "option_type": "{d["option_type"]}"\n'
        f'  "K": {d["strike"]}\n'
        f'  "T": {d["T_approx"]}\n'
        f"  (ticker={d['ticker']}, expiry={d['expiry']}, today={date.today().isoformat()})\n"
        f"ONLY fill: S (spot price for {d['ticker']}, default 100 if unknown) "
        f"and sigma (default 0.2). Use model=bsm unless user says otherwise."
    )

SYSTEM_PROMPT = """You are an option pricing parameter extractor. Given a user's natural language description of an option, extract the pricing parameters and select the appropriate pricing model.

## Model Selection Rules
- If the user only provides a volatility parameter (sigma/σ/波动率), select **BSM**.
- If the user provides beta (β/弹性参数) or mentions "CEV", select **CEV**.
- If the user provides kappa/κ, theta/θ, xi/ξ, rho/ρ, v0 (initial variance), or mentions "Heston", "stochastic volatility", or "随机波动率", select **Heston**.
- Default to **BSM** when no specific model is indicated.

## Parameter Defaults
Use the following defaults for any parameter the user does not specify:
- S = 100 (underlying price)
- K = 100 (strike price)
- T = 1.0 (time to expiry in years)
- r = 0.05 (risk-free rate)
- sigma = 0.2 (volatility for BSM/CEV)
- beta = 0.5 (elasticity for CEV)
- v0 = 0.04 (initial variance for Heston)
- kappa = 2.0 (mean reversion speed for Heston)
- theta = 0.04 (long-term variance for Heston)
- xi = 0.3 (vol-of-vol for Heston)
- rho = -0.7 (correlation for Heston)

## Output Format
Return ONLY a valid JSON object with the following schema, no additional text:
{
  "model": "bsm" | "cev" | "heston",
  "option_type": "call" | "put",
  "S": float,
  "K": float,
  "T": float,
  "r": float,
  "sigma": float,
  "beta": float,
  "v0": float,
  "kappa": float,
  "theta": float,
  "xi": float,
  "rho": float
}

## Examples
Input: "price a 6-month European call on TSLA at $250 strike, spot $260, 30% vol, rate 5%"
Output: {"model":"bsm","option_type":"call","S":260,"K":250,"T":0.5,"r":0.05,"sigma":0.3,"beta":1.0,"v0":0.09,"kappa":0.0,"theta":0.0,"xi":0.0,"rho":0.0}

Input: "Heston看跌期权，S=100，K=105，T=1年，r=0.05，kappa=2，theta=0.04，xi=0.3，rho=-0.7，v0=0.04"
Output: {"model":"heston","option_type":"put","S":100,"K":105,"T":1.0,"r":0.05,"sigma":0.0,"beta":1.0,"v0":0.04,"kappa":2.0,"theta":0.04,"xi":0.3,"rho":-0.7}
"""

# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def validate_params(params: dict) -> dict:
    """Validate and clamp extracted parameters to legal ranges.

    Returns corrected params dict with warnings printed for any fix-ups.
    """
    checks = [
        ("S",     1e-2,  1e6),
        ("K",     1e-2,  1e6),
        ("T",     0.001, 20.0),
        ("r",     -0.1,  0.5),
        ("sigma", 0.0,   5.0),
        ("beta",  0.01,  1.0),
        ("v0",    1e-6,  5.0),
        ("kappa", 0.0,   50.0),
        ("theta", 1e-6,  5.0),
        ("xi",    0.0,   5.0),
        ("rho",   -0.999, 0.999),
    ]
    # parameters only relevant for Heston (their defaults of 0 are fine for BSM/CEV)
    _heston_keys = {"kappa", "theta", "xi", "rho", "v0"}

    for key, lo, hi in checks:
        if key not in params:
            continue
        orig = params[key]
        params[key] = float(max(lo, min(hi, orig)))
        if params[key] != orig and key not in _heston_keys:
            print(f"[WARN] {key}={orig:.4f} out of range [{lo},{hi}], clamped to {params[key]:.4f}")
        elif params[key] != orig and params.get("model") == "heston":
            print(f"[WARN] {key}={orig:.4f} out of range [{lo},{hi}], clamped to {params[key]:.4f}")

    if params["model"] not in ("bsm", "cev", "heston"):
        print(f"[WARN] unknown model '{params['model']}', defaulting to bsm")
        params["model"] = "bsm"
    if params["option_type"] not in ("call", "put"):
        print(f"[WARN] unknown option_type '{params['option_type']}', defaulting to call")
        params["option_type"] = "call"

    return params


# ---------------------------------------------------------------------------
# LLM Router
# ---------------------------------------------------------------------------

class LLMRouter:
    """DeepSeek-based natural language interface for option pricing.

    Extracts model type, option parameters, and delegates to UnifiedPINN
    for pricing.  Torch/PINN imports are lazy so that extract_params() works
    in environments without PyTorch installed.
    """

    def __init__(self, api_key: str = DEEPSEEK_API_KEY, device=None):
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self._device = device  # None means auto-detect on first PINN load
        self._pinn = None      # lazy init
        self._ModelParams = None

    @property
    def pinn(self):
        """Lazy-load UnifiedPINN with the trained checkpoint."""
        if self._pinn is None:
            self._pinn = self._load_pinn()
        return self._pinn

    @property
    def ModelParams(self):
        if self._ModelParams is None:
            from unified_pinn_v2 import ModelParams
            self._ModelParams = ModelParams
        return self._ModelParams

    def _get_device(self):
        if self._device is not None:
            return self._device
        import torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def _load_pinn(self):
        from unified_pinn_v2 import UnifiedPINN
        ModelParams = self.ModelParams
        device = self._get_device()
        ckpt_path = os.path.join(BASE, "results", "unified_v16_gl.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        param_list = self._build_param_list()
        pinn = UnifiedPINN(param_list, hidden=128, depth=6, device=device)
        pinn.load(ckpt_path)
        pinn.net.eval()
        return pinn

    def _build_param_list(self):
        ModelParams = self.ModelParams
        params = []
        for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
            params.append(ModelParams.from_bsm(sigma=sigma))
        for sigma in [0.15, 0.2, 0.25]:
            for beta in [0.3, 0.5, 0.7, 0.9]:
                params.append(ModelParams.from_cev(sigma=sigma, beta=beta))
        for kappa in [1.0, 2.0, 3.0]:
            for theta in [0.02, 0.04, 0.06]:
                for xi in [0.2, 0.3, 0.4]:
                    for rho in [-0.7, -0.5]:
                        params.append(ModelParams.from_heston(
                            kappa=kappa, theta=theta, xi=xi, rho=rho, v0=theta))
        return params

    def extract_params(self, user_input: str, retry: bool = True) -> dict:
        """Send user input to DeepSeek and return parsed parameter dict.

        Automatically detects and decodes OCC option symbols before LLM call.
        Raises ValueError on JSON parse failure after retry.
        """
        # Preprocess: detect OCC symbol
        user_text = user_input.strip()
        if is_occ_symbol(user_text):
            user_text = _format_occ_for_llm(user_text)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ]
        raw = ""
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=512,
                )
                raw = resp.choices[0].message.content.strip()
                params = json.loads(raw)
                return params
            except json.JSONDecodeError:
                if attempt == 0 and retry:
                    continue
                raise ValueError(f"Failed to parse LLM JSON response.\nRaw: {raw}")
            except Exception as e:
                if attempt == 0 and retry:
                    continue
                raise ValueError(f"LLM API call failed: {e}")

    def price(self, user_input: str) -> dict:
        """Full pipeline: natural language -> params -> validate -> PINN price.

        Returns a dict with extracted params, model selection, and price.
        """
        params = self.extract_params(user_input)
        params = validate_params(params)
        ModelParams = self.ModelParams

        model_type   = params["model"]
        option_type  = params["option_type"]
        S            = params["S"]
        K            = params["K"]
        T            = params["T"]
        r            = params["r"]
        sigma        = params["sigma"]
        beta         = params["beta"]
        kappa        = params["kappa"]
        theta        = params["theta"]
        xi           = params["xi"]
        rho          = params["rho"]
        v0           = params["v0"]

        if model_type == "bsm":
            p = ModelParams.from_bsm(K=K, T=T, r=r, sigma=sigma,
                                     option_type=option_type)
        elif model_type == "cev":
            p = ModelParams.from_cev(K=K, T=T, r=r, sigma=sigma, beta=beta,
                                     option_type=option_type)
        else:  # heston
            p = ModelParams.from_heston(K=K, T=T, r=r,
                                        kappa=kappa, theta=theta,
                                        xi=xi, rho=rho, v0=v0,
                                        option_type=option_type)

        price_val = self.pinn.price(p, S=S, v=v0, t=0.0)

        return {
            "model":       model_type,
            "option_type": option_type,
            "S":           S,
            "K":           K,
            "T":           T,
            "r":           r,
            "sigma":       sigma,
            "beta":        beta,
            "kappa":       kappa,
            "theta":       theta,
            "xi":          xi,
            "rho":         rho,
            "v0":          v0,
            "price":       round(price_val, 6),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM + PINN unified option pricing -- natural language input"
    )
    parser.add_argument(
        "query", type=str,
        help="Natural language option description or OCC symbol, e.g. "
             "'定价一份欧式看涨期权 TSLA S=250 K=260 T=0.5 sigma=0.3' or "
             "'AAPL230616C00150000'"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON (no formatting)"
    )
    args = parser.parse_args()

    router = LLMRouter()
    try:
        result = router.price(args.query)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        opt_label = "看涨" if result["option_type"] == "call" else "看跌"
        model_label = {"bsm": "BSM", "cev": "CEV", "heston": "Heston"}[result["model"]]
        print(f"{'='*60}")
        print(f"  模型: {model_label}")
        print(f"  类型: {opt_label}期权 (European {result['option_type']})")
        print(f"  标的现价 S: {result['S']}")
        print(f"  行权价 K:   {result['K']}")
        print(f"  到期时间 T: {result['T']} 年")
        print(f"  无风险利率 r: {result['r']}")
        if result["model"] in ("bsm", "cev"):
            print(f"  波动率 sigma: {result['sigma']}")
        if result["model"] == "cev":
            print(f"  弹性参数 beta: {result['beta']}")
        if result["model"] == "heston":
            print(f"  kappa={result['kappa']}, theta={result['theta']}, "
                  f"xi={result['xi']}, rho={result['rho']}, v0={result['v0']}")
        print(f"  {'-'*40}")
        print(f"  期权价格: {result['price']:.6f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
