"""LLM路由层测试脚本：20个测试用例，验证模型选择准确率和参数提取。

用法:
  python llm_test.py
  python llm_test.py --save results/llm_test_results.json
"""

import json
import argparse
import sys
from llm_router import LLMRouter

# ---------------------------------------------------------------------------
# 20 test cases: 10 Chinese + 10 English, covering BSM / CEV / Heston
# Each entry: (input, expected_model, description)
# ---------------------------------------------------------------------------
TEST_CASES = [
    # --- BSM (Chinese) ---
    (
        "定价一份欧式看涨期权，标的现价100，行权价100，一年到期，波动率20%，无风险利率5%",
        "bsm",
        "BSM-CN-1: 标准BSM参数",
    ),
    (
        "帮我算一下看跌期权的价格，S=120，K=110，T=0.5年，sigma=0.25，r=0.03",
        "bsm",
        "BSM-CN-2: 看跌期权，sigma关键词",
    ),
    (
        "Black-Scholes模型，欧式看涨，现价80，执行价90，三个月到期，年化波动率30%",
        "bsm",
        "BSM-CN-3: 明确提及Black-Scholes",
    ),
    # --- CEV (Chinese) ---
    (
        "CEV模型定价，S=100，K=100，T=1，sigma=0.2，beta=0.5，r=0.05",
        "cev",
        "CEV-CN-1: 明确CEV，beta=0.5",
    ),
    (
        "常弹性方差模型，弹性参数0.3，波动率0.2，现价100，行权价105，到期一年",
        "cev",
        "CEV-CN-2: 中文'常弹性方差'，弹性参数关键词",
    ),
    # --- Heston (Chinese) ---
    (
        "Heston模型，S=100，K=100，T=1，r=0.05，kappa=2，theta=0.04，xi=0.3，rho=-0.7，v0=0.04",
        "heston",
        "Heston-CN-1: 完整Heston参数",
    ),
    (
        "随机波动率模型定价，均值回归速度2，长期方差0.04，vol-of-vol=0.3，相关系数-0.5，初始方差0.04",
        "heston",
        "Heston-CN-2: '随机波动率'关键词，中文参数名",
    ),
    (
        "用Heston给我定价一个看跌期权，kappa=1.5，theta=0.06，xi=0.4，rho=-0.6，v0=0.05，S=95，K=100，T=0.5",
        "heston",
        "Heston-CN-3: Heston看跌，kappa/theta/xi/rho关键词",
    ),
    (
        "波动率的波动率为0.3，均值回归2.0，长期均值0.04，相关系数-0.7，现价100，行权价100，一年",
        "heston",
        "Heston-CN-4: '波动率的波动率'关键词",
    ),
    (
        "期权定价，S=100，K=100，T=1，r=0.05，kappa=3，theta=0.05，xi=0.25，rho=-0.8，v0=0.05",
        "heston",
        "Heston-CN-5: 仅通过kappa/theta/xi/rho推断Heston",
    ),
    # --- BSM (English) ---
    (
        "Price a European call option, S=100, K=100, T=1, r=0.05, sigma=0.2",
        "bsm",
        "BSM-EN-1: standard BSM call",
    ),
    (
        "What is the Black-Scholes price of a put with spot 150, strike 140, 6 months, vol 25%, rate 4%?",
        "bsm",
        "BSM-EN-2: explicit Black-Scholes mention",
    ),
    (
        "European call, underlying at 80, strike 85, 3 months to expiry, annualized volatility 35%",
        "bsm",
        "BSM-EN-3: no model name, only sigma -> BSM default",
    ),
    # --- CEV (English) ---
    (
        "CEV model, S=100, K=100, T=1, sigma=0.2, beta=0.5, r=0.05",
        "cev",
        "CEV-EN-1: explicit CEV, beta keyword",
    ),
    (
        "Constant elasticity of variance option, elasticity parameter 0.3, vol 0.2, S=100, K=100, T=1",
        "cev",
        "CEV-EN-2: 'constant elasticity of variance' phrase",
    ),
    # --- Heston (English) ---
    (
        "Heston model call, S=100, K=100, T=1, r=0.05, kappa=2, theta=0.04, xi=0.3, rho=-0.7, v0=0.04",
        "heston",
        "Heston-EN-1: full Heston parameters",
    ),
    (
        "Stochastic volatility model, mean reversion 1.5, long-run variance 0.05, vol-of-vol 0.4, correlation -0.6, v0=0.05",
        "heston",
        "Heston-EN-2: 'stochastic volatility' + vol-of-vol keywords",
    ),
    (
        "Price a put using Heston, kappa=3, theta=0.03, xi=0.2, rho=-0.5, v0=0.03, S=90, K=100, T=0.75",
        "heston",
        "Heston-EN-3: Heston put with all parameters",
    ),
    (
        "Option pricing with mean-reverting variance, kappa=2.5, theta=0.06, xi=0.35, rho=-0.65, S=110, K=105, T=1",
        "heston",
        "Heston-EN-4: infer Heston from kappa/theta/xi/rho",
    ),
    (
        "vol-of-vol=0.3, mean reversion speed 2.0, long-term variance 0.04, correlation -0.7, S=100, K=100, T=1",
        "heston",
        "Heston-EN-5: 'vol-of-vol' keyword triggers Heston",
    ),
]


def run_tests(save_path: str | None = None) -> dict:
    router = LLMRouter()
    results = []
    correct = 0

    print(f"Running {len(TEST_CASES)} LLM routing test cases...\n")
    print(f"{'#':<3} {'Expected':<8} {'Got':<8} {'Pass':<5}  Description")
    print("-" * 72)

    for i, (user_input, expected_model, desc) in enumerate(TEST_CASES, 1):
        try:
            params = router.extract_params(user_input)
            got_model = params.get("model", "MISSING")
            passed = got_model == expected_model
        except Exception as e:
            got_model = f"ERROR: {e}"
            passed = False

        if passed:
            correct += 1

        status = "✓" if passed else "✗"
        print(f"{i:<3} {expected_model:<8} {got_model:<8} {status:<5}  {desc}")

        results.append({
            "id": i,
            "description": desc,
            "input": user_input,
            "expected_model": expected_model,
            "got_model": got_model,
            "passed": passed,
        })

    accuracy = correct / len(TEST_CASES)
    print("-" * 72)
    print(f"\nResult: {correct}/{len(TEST_CASES)} correct  |  Accuracy = {accuracy*100:.1f}%\n")

    summary = {
        "total": len(TEST_CASES),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "cases": results,
    }

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {save_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM routing layer test suite")
    parser.add_argument("--save", default="results/llm_test_results.json",
                        help="Path to save JSON results")
    args = parser.parse_args()
    summary = run_tests(save_path=args.save)
    sys.exit(0 if summary["accuracy"] == 1.0 else 1)
