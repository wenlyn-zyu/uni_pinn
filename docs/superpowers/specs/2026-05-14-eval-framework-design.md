# 统一评估框架设计

**日期**: 2026-05-14  
**项目**: uni_pinn — 基于PINN的期权定价（BSM/CEV/Heston）  
**目标**: 清理代码、建立统一评估框架、更新论文实验结果

---

## 背景与目标

论文主线是 unified_pinn_v2（单网络同时处理BSM/CEV/Heston三个模型），三个独立PINN作为 related work 基线，参数化PINN作为 future work。

需要在同一套数据上评估所有模型，输出可直接引用到论文的结果表格。

---

## 一、目录清理

### 删除文件（`option_pinn/` 下）

| 文件 | 原因 |
|------|------|
| `spy_backtest_v15.py` | 旧版本，被 eval_all.py 取代 |
| `spy_backtest_v15_ft2.py` | 旧版本 |
| `spy_backtest_parametric.py`（根目录） | 重复文件 |
| `train_v15_schr.py` | 训练脚本，checkpoint 已保存 |
| `train_v16_cont.py` | 训练脚本，checkpoint 已保存 |
| `train_v16_heston_gl.py` | 训练脚本，checkpoint 已保存 |
| `eval_compare.py` | 被 eval_all.py 取代 |
| `evaluate.py` | 被 eval_all.py 取代 |
| `parametric_pinn/quick_test.py` | 临时测试脚本 |
| `parametric_pinn/generate_ref_data.py` | 数据已生成 |
| `parametric_pinn/generate_ref_fast.py` | 数据已生成 |

### 保留文件

- `independent/` 下所有模型文件（bsm_pinn.py, cev_pinn.py, heston_pinn.py 等）
- `unified_pinn_v2.py`
- `parametric_pinn/fully_parametric_pinn.py`, `train_parametric.py`, `evaluate_parametric.py`
- `greeks_validation.py`（复用解析解函数）
- `data/spy_quotedata.csv`

---

## 二、新增文件结构

```
option_pinn/
├── eval_all.py                    # 统一评估入口（新建）
├── finetune_heston.py             # unified_v2 Heston fine-tune（新建）
└── results/
    ├── eval_synthetic.csv         # 合成数据评估结果
    ├── eval_market.csv            # 真实市场数据评估结果
    └── eval_greeks.csv            # Greeks 精度汇总
```

---

## 三、模型与 Checkpoint 对应关系

| 模型名称 | Checkpoint 路径 | 论文定位 |
|---------|----------------|---------|
| BSM 独立PINN | `results/indep_bsm.pt` | Related work |
| CEV 独立PINN | `results/indep_cev.pt` | Related work |
| Heston 独立PINN | `results/indep_heston.pt` | Related work |
| Unified PINN v2 | `results/unified_v16_gl.pt` | **主线** |
| Unified v2 fine-tuned | `results/unified_v2_ft.pt`（新生成） | **主线补充** |
| 参数化PINN | `parametric_pinn/results/fully_param_v1.pt` | Future work |

---

## 四、合成数据评估

**目的**：验证训练拟合效果（in-distribution）

### 数据生成（固定参数网格）

| 模型 | 固定参数 | S 网格 | K | T | r |
|------|---------|--------|---|---|---|
| BSM | σ=0.2 | linspace(50, 250, 50) | 100 | 1.0 | 0.05 |
| CEV | σ=0.25, β=0.5 | linspace(50, 250, 50) | 100 | 1.0 | 0.05 |
| Heston | κ=2.0, θ=0.04, ξ=0.3, ρ=-0.7, v₀=0.04 | linspace(50, 250, 50) | 100 | 1.0 | 0.05 |

参数与各独立PINN训练时完全一致，确保验证的是拟合能力而非泛化能力。

### 参考解

- BSM → Black-Scholes 解析解（`bs_call_price()`）
- CEV → Schroder (1989) 非中心卡方分布公式（`cev_analytical_call()`，scipy.stats.ncx2）
- Heston → Gauss-Legendre 96点特征函数半解析解（`heston_price_gl()`）

### 评估指标

- **价格**：MSE、RelMSE（相对均方误差）
- **Greeks**（autograd 计算，与解析解对比）：
  - BSM/CEV：Delta (dV/dS)、Gamma (d²V/dS²)
  - Heston：Delta、Gamma、Vega (dV/dv)

---

## 五、真实市场数据评估

**目的**：验证训练后模型的应用效果（out-of-distribution）

### 数据处理

- 来源：`data/spy_quotedata.csv`（SPY 期权链，约6450条）
- 目标价格：mid-price = `(bid + ask) / 2`
- 过滤：去除 bid=0 或 ask=0 的无效报价，去除深度实值/虚值（moneyness 0.7~1.3）

### 参与评估的模型

1. BSM 独立PINN
2. CEV 独立PINN
3. Heston 独立PINN
4. Unified PINN v2（原始）
5. Unified PINN v2 fine-tuned（Heston分支）
6. 参数化PINN

### 参考解

同合成数据，用解析解/半解析解计算理论价格，与市场 mid-price 对比。

---

## 六、Fine-tune 方案（`finetune_heston.py`）

**背景**：unified_pinn_v2 在真实市场数据上 Heston 分支表现可能不佳，需要域适应。

**方案**：
- 加载 `results/unified_v16_gl.pt`
- 在 SPY 数据中筛选适合 Heston 定价的合约（隐含波动率微笑明显的）
- 用市场 mid-price 作为数据锚点（data loss）
- 保留 PDE loss（防止物理约束退化）
- 只更新网络中与 Heston 参数相关的部分（或全量 fine-tune，lr 降低10倍）
- 保存为 `results/unified_v2_ft.pt`

**Loss 设计**：
```
L_total = w_pde * L_pde + w_data * L_data
w_pde = 0.1（降低，避免压制数据信号）
w_data = 1.0
```

---

## 七、输出格式

### `eval_synthetic.csv`

```
model, metric_type, bsm_mse, bsm_relmse, cev_mse, cev_relmse, heston_mse, heston_relmse
indep_bsm, price, ...
indep_cev, price, ...
indep_heston, price, ...
unified_v2, price, ...
parametric, price, ...
```

### `eval_greeks.csv`

```
model, model_type, delta_mae, gamma_mae, vega_mae
indep_bsm, BSM, ...
unified_v2, BSM, ...
...
```

### `eval_market.csv`

```
model, mse, relmse, delta_mae, gamma_mae, vega_mae
indep_bsm, ...
unified_v2, ...
unified_v2_ft, ...
...
```

---

## 八、论文更新方向

| 章节 | 内容 |
|------|------|
| 第2章（Related Work） | 三个独立PINN复现，引用 eval_synthetic 结果作为基线 |
| 第3-4章（主线） | unified_pinn_v2 设计、训练、合成+市场评估结果 |
| 第4章（实验） | 对比表格：独立PINN vs unified_v2 vs unified_v2_ft |
| 第5章（Future Work） | 参数化PINN，说明泛化局限，指出改进方向 |

---

## 九、执行顺序

1. 清理冗余文件（本地 + 服务器）
2. 编写 `eval_all.py`（合成数据评估）
3. 编写 `finetune_heston.py`，在服务器上运行 fine-tune
4. 扩展 `eval_all.py` 加入真实市场数据评估
5. git 同步本地与服务器
6. 更新论文 LaTeX 中的实验表格和分析文字
