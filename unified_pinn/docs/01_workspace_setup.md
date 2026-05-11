# 01 — 工作区搭建与统一PINN核心设计

**完成时间**：2026-05-09  
**对应文件**：`unified_pinn.py`、`train.py`、`evaluate.py`

---

## 你做了什么

在服务器 `~/zhuwl2022/unified_pinn/` 下建立了一个独立实验工作区，包含三个文件：

```
unified_pinn/
├── unified_pinn.py   # 核心：统一PDE算子 + 网络 + 训练器
├── train.py          # 训练入口（命令行参数）
├── evaluate.py       # 精度评估 + 绘图
├── docs/             # 本文档所在目录
└── results/          # 训练输出（权重、日志、图表）
```

这个工作区和原始仓库 `ppin/` 完全独立，不影响已有代码。

---

## 为什么这么做：核心思想

### 原始方案的问题

原始 `ppin/src/models/` 里有三个独立文件：`bsm_pinn.py`、`cev_pinn.py`、`heston_pinn.py`。  
每个文件训练一个独立的网络，LLM 路由层负责"选择"用哪个。

这个方案的局限：
- 三个网络各自训练，参数量 × 3，推理时要加载三套权重
- LLM 路由层需要做离散的模型选择，一旦选错就完全错误
- 无法处理"中间状态"（比如弱随机波动率，介于 CEV 和 Heston 之间）

### 统一方案的核心观察

三个模型的 PDE 算子可以写成同一个形式：

```
F[V] = V_t
     + 0.5 * a(S,v;λ) * V_SS     ← S方向扩散
     + b(S,v;λ)       * V_Sv     ← 交叉项（BSM/CEV中为0）
     + 0.5 * c(v;λ)   * V_vv     ← v方向扩散（BSM/CEV中为0）
     + r*S             * V_S
     + d(v;λ)          * V_v     ← v漂移（BSM/CEV中为0）
     - r*V = 0
```

系数由参数向量 `λ = (σ, β, κ, θ, ξ, ρ)` 决定：

| 系数 | BSM | CEV | Heston |
|------|-----|-----|--------|
| `a`  | σ²S² | σ²S^(2β) | vS² |
| `b`  | 0 | 0 | ρξvS |
| `c`  | 0 | 0 | ξ²v |
| `d`  | 0 | 0 | κ(θ-v) |

**BSM 是 CEV 在 β=1 时的特例；当 ξ=0 时 Heston 退化为 BSM/CEV。**  
三者之间存在参数空间上的连续性，可以用一个网络统一学习。

---

## 三个关键设计决策

### 1. 网络输入扩展到9维

原始 BSM 网络输入是 `[S_n, t_n]`（2维）。  
统一网络输入是 `[S_n, v_n, t_n, σ, β, κ, θ, ξ, ρ]`（9维）。

参数向量 λ 从"训练时固定的超参数"变成了"运行时的网络输入"。  
这意味着：**同一个网络，输入不同的 λ，就能给出不同模型的解**。

```python
# 原始方式：sigma 是类属性，训练时固定
class BSM_PINN:
    def __init__(self, sigma=0.2, ...):
        self.sigma = sigma   # 固定死了

# 统一方式：sigma 是网络输入，推理时传入
model.price(ModelParams.from_bsm(sigma=0.2), S=100.)
model.price(ModelParams.from_bsm(sigma=0.3), S=100.)  # 同一个网络，不同sigma
```

### 2. 软掩码（Soft Mask）实现连续退化

BSM/CEV 和 Heston 的 S 方向扩散系数形式不同（`σ²S^(2β)` vs `vS²`）。  
如果用 if-else 切换，梯度会在边界处不连续，训练不稳定。

解决方案：用 ξ 作为软开关：

```python
mask = tanh(xi / 0.05) ** 2   # xi=0 → mask=0；xi大 → mask≈1
a = (1 - mask) * sigma**2 * S**(2*beta) + mask * v * S**2
```

- 当 `ξ=0`（BSM/CEV）：`mask=0`，`a = σ²S^(2β)`，交叉项和v扩散项也自然为0
- 当 `ξ=0.3`（Heston）：`mask≈1`，`a = vS²`，完整Heston算子生效
- 中间值：平滑插值，梯度连续

### 3. 混合批次训练

每个 epoch 从三个模型各采 5000 个配点，拼成 15000 点的混合批次：

```python
# 三个模型的配点拼在一起，一次反向传播
S_c, v_c, t_c, lam_c = self._sample_batch(n_per_model=5000)
# lam_c 的前5000行是BSM的λ，中间5000行是CEV的λ，后5000行是Heston的λ
res = unified_pde_residual(net, S_c, v_c, t_c, lam_c, ...)
loss_pde = mean(res**2)
```

这样网络在每次更新时都同时收到三个模型的监督信号，不会"遗忘"某个模型。

### 4. 硬约束（ICPINN）统一应用

原始代码只有 Heston 用了硬约束，BSM/CEV 用软约束（损失函数里加 L_ic 项）。  
统一网络对所有模型都用硬约束：

```python
V = payoff(S) + (T - t) * net(S_n, v_n, t_n, lam)
```

当 `t=T` 时，`(T-t)=0`，所以 `V(S,v,T) = payoff(S)` 精确成立，无需 L_ic 项。  
这减少了一个损失权重超参数（`w_ic`），训练更稳定。

---

## 如何逐步运行

### 前提：SSH 连接到服务器

```bash
ssh idata2
# 或者完整写法：ssh yz2026@10.19.126.135
```

### 第一步：激活环境，进入工作区

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pinn_option
cd ~/zhuwl2022/unified_pinn
```

### 第二步：快速验证代码可以运行

```bash
python -c "
from unified_pinn import ModelParams, UnifiedPINN
p = ModelParams.from_bsm()
model = UnifiedPINN([p])
print('设备:', model.device)
print('参数量:', sum(p.numel() for p in model.net.parameters()))
print('前向测试:', model.price(p, S=100.))
"
```

期望输出：
```
设备: cuda
参数量: 83969
前向测试: <某个数值，训练前是随机的>
```

### 第三步：启动训练（后台运行）

```bash
mkdir -p results
nohup python train.py \
    --epochs 30000 \
    --n_per_model 5000 \
    --out results/unified.pt \
    > results/train.log 2>&1 &
echo "PID: $!"
```

参数说明：
- `--epochs 30000`：训练轮数，约20分钟（RTX PRO 6000）
- `--n_per_model 5000`：每个模型每轮采样点数，总批次=15000
- `--out`：模型权重保存路径

### 第四步：监控训练进度

```bash
# 查看最新损失
tail -f results/train.log | grep -E "loss=|epoch"

# 或者查看进度条快照
tail -1 results/train.log
```

正常的损失下降曲线：
- epoch 1000：loss ≈ 1e-1
- epoch 10000：loss ≈ 1e-2
- epoch 30000：loss ≈ 1e-3 ~ 1e-4

### 第五步：训练完成后评估

```bash
python evaluate.py \
    --ckpt results/unified.pt \
    --out results/
```

输出示例：
```
=== 评估结果 ===
  BSM    MAE=0.0312  RMSE=0.0401  MaxErr=0.1023  RelMAE=0.0089
  CEV    MAE=0.0445  RMSE=0.0567  MaxErr=0.1234  RelMAE=0.0121
  Heston MAE=0.0521  RMSE=0.0678  MaxErr=0.1456  RelMAE=0.0143
图表已保存至 results/unified_eval.pdf
```

### 第六步：用统一网络推断任意参数

```python
from unified_pinn import ModelParams, UnifiedPINN
import torch

# 加载训练好的模型
p_bsm    = ModelParams.from_bsm()
p_cev    = ModelParams.from_cev()
p_heston = ModelParams.from_heston()

model = UnifiedPINN([p_bsm, p_cev, p_heston])
model.load("results/unified.pt")

# 同一个网络，三种模型
print(model.price(p_bsm,    S=100.))   # BSM定价
print(model.price(p_cev,    S=100.))   # CEV定价
print(model.price(p_heston, S=100.))   # Heston定价

# 还可以用训练时没见过的参数（泛化能力）
p_new = ModelParams.from_bsm(sigma=0.35, K=110.)
print(model.price(p_new, S=105.))
```

---

## 与原始代码的对比

| 维度 | 原始方案（ppin/） | 统一方案（unified_pinn/） |
|------|-----------------|------------------------|
| 网络数量 | 3个独立网络 | 1个网络 |
| 网络输入维度 | 2（BSM/CEV）或3（Heston） | 9（统一） |
| 模型切换 | LLM路由层离散选择 | 参数向量连续控制 |
| 终值条件 | 软约束（BSM/CEV）+ 硬约束（Heston） | 全部硬约束 |
| 泛化能力 | 仅限训练时的固定参数 | 参数空间内连续插值 |
| 训练时间 | 3 × 20min = 60min | ~20min |

---

## 已知局限（待后续文档解决）

1. **K/T 在混合批次中共享**：当前实现里 `unified_pde_residual` 的 K、T 取第一个模型的值，三个模型必须用相同的 K 和 T。后续可以把 K、T 也加入 λ 向量。

2. **BSM/CEV 的 v 维度是冗余的**：对 BSM/CEV，v 不是状态变量，但网络仍然接受 v 作为输入。训练时 v 从 `[1e-4, v_max]` 均匀采样，网络需要学会"忽略"这个维度。可以通过在 BSM/CEV 的配点中固定 `v = sigma²` 来改善。

3. **软掩码的阈值 0.05 是经验值**：如果 ξ 的训练范围变化，可能需要调整。
