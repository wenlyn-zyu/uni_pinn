# 02 — 精度诊断与残差硬约束修复

**完成时间**：2026-05-09  
**对应文件**：`unified_pinn.py`（修订版 v2）  
**训练输出**：`results/unified_v2.pt`、`results/train_v2.log`

---

## 你做了什么

发现并修复了 v1 版本的核心精度问题：

- **v1 问题**：BSM 在 S=100 处误差 -7.8（真实值 10.45，预测值 2.65），且单模型 5000 epoch 和混合 30000 epoch 误差完全相同，说明网络陷入了局部最优，不是训练不足。
- **v2 修复**：将硬约束从 `payoff + (T-t)*net` 改为 `V_bs + (T-t)*net`，让网络只学习相对于 BS 解析解的残差修正量。
- **验证**：未训练的 v2 网络在 BSM 上误差已经从 -7.8 降到 +0.05，基线正确。

---

## 为什么这么做：根本原因分析

### v1 的问题在哪里

v1 的硬约束是：

```
V(S, v, t) = payoff(S) + (T - t) * net(S_n, v_n, t_n, λ)
```

对于 ATM 期权（S=K=100，t=0，T=1）：
- `payoff(100) = max(100-100, 0) = 0`
- 所以 `V(100, v, 0) = 1.0 * net(...)`
- 要让 V=10.45，net 必须输出 10.45

**问题**：网络用 Tanh 激活，最后一层是线性层，初始化后输出接近 0。PDE 残差对 net 的输出量级没有直接约束——PDE 残差为零的条件是"满足微分方程"，而不是"输出正确的绝对值"。

结果：网络找到了一个 PDE 残差很小但绝对值完全错误的解（net≈2.65，而非 10.45）。这是 PINN 的经典"谱偏差"问题：网络倾向于学习低频、小幅度的函数。

### v2 的修复思路

改用 BS 解析解作为基线：

```
V(S, v, t) = V_bs(S, τ; σ_eff) + τ * net(S_n, v_n, t_n, λ)
```

其中：
- `τ = T - t`（剩余时间）
- `σ_eff`：等效波动率，用软掩码从 λ 中计算
  - BSM/CEV：`σ_eff = σ`
  - Heston：`σ_eff = √v`
- `V_bs`：用 `σ_eff` 计算的 BS 解析解（精确公式）

**好处**：
1. 初始化时 `net≈0`，所以 `V≈V_bs`，已经是正确量级的期权价格
2. 网络只需学习"BS 近似解和真实解之间的差"，这个差通常很小（对 BSM 为零，对 CEV/Heston 是修正项）
3. 梯度信号充足：`∂L/∂net` 不再需要把 net 从 0 推到 10，而是从 0 推到一个小的修正量

**对三个模型的含义**：
- BSM：`V_bs` 精确，`net` 应该趋近于 0（学习数值误差）
- CEV（β≠1）：`V_bs` 是 β=1 的近似，`net` 学习弹性指数修正
- Heston：`V_bs` 是确定性波动率近似，`net` 学习随机波动率修正

### 等效波动率的计算

```python
mask = tanh(xi / 0.05)^2          # xi=0 → mask=0 (BSM/CEV)
sigma_eff = (1-mask)*sigma + mask*sqrt(v)
```

- BSM（xi=0）：`mask=0`，`sigma_eff = sigma = 0.2` ✓
- Heston（xi=0.3）：`mask≈1`，`sigma_eff = sqrt(v) = sqrt(0.04) = 0.2` ✓（v0=theta=0.04 时和 BSM 一致）

---

## 如何逐步运行

### 第一步：验证修复效果（未训练基线）

```bash
ssh idata2
source ~/anaconda3/etc/profile.d/conda.sh && conda activate pinn_option
cd ~/zhuwl2022/unified_pinn

python - << 'EOF'
from unified_pinn import ModelParams, UnifiedPINN
from evaluate import bs_call_price

p_bsm = ModelParams.from_bsm()
model = UnifiedPINN([p_bsm])   # 未训练

print('S    | PINN(未训练) | BSM   | diff')
for S in [80, 90, 100, 110, 120]:
    pred = model.price(p_bsm, S=float(S))
    ref  = bs_call_price(S, 100, 1.0, 0.05, 0.2)
    print('{:4d} | {:11.3f} | {:5.3f} | {:+.3f}'.format(S, pred, ref, pred-ref))
EOF
```

期望输出（误差应在 ±0.1 以内）：
```
S    | PINN(未训练) | BSM   | diff
  80 |       1.910 | 1.859 | +0.050
  90 |       5.142 | 5.091 | +0.050
 100 |      10.501 | 10.451 | +0.050
 110 |      17.713 | 17.663 | +0.050
 120 |      26.220 | 26.169 | +0.051
```

如果这一步输出的 diff 很大（>1），说明 `sigma_eff` 计算有问题，不要继续训练。

### 第二步：启动 v2 训练

```bash
nohup python train.py \
    --epochs 30000 \
    --n_per_model 5000 \
    --out results/unified_v2.pt \
    > results/train_v2.log 2>&1 &
echo "PID: $!"
```

### 第三步：监控训练

```bash
# 实时查看损失
tail -f results/train_v2.log | grep -E "loss=|pde="

# 查看最新几条记录
tail -3 results/train_v2.log
```

v2 的损失应该比 v1 下降更快，因为网络从正确量级开始优化：
- epoch 500：loss ≈ 1e-3（v1 是 5e-1）
- epoch 5000：loss ≈ 1e-4
- epoch 30000：loss ≈ 1e-5 ~ 1e-6（目标）

### 第四步：训练完成后评估

```bash
python evaluate.py --ckpt results/unified_v2.pt --out results/
```

目标精度（MAE）：
- BSM：< 0.05（约 0.5% 相对误差）
- CEV：< 0.1
- Heston：< 0.2

---

## v1 vs v2 对比

| 指标 | v1（payoff 硬约束） | v2（BS 残差硬约束） |
|------|-------------------|-------------------|
| 未训练 BSM 误差（S=100） | -7.8 | +0.05 |
| 30000 epoch 后 BSM MAE | 1.90 | 待测 |
| 训练速度 | ~17 it/s | ~12 it/s（BS 计算开销） |
| 收敛难度 | 高（需要从 0 学到 10） | 低（从 0 学到小修正量） |

---

## 已知局限

1. **BS 基线对 Heston 是近似**：当随机波动率很强（xi 大、rho 极端）时，BS 近似误差大，net 需要学习的修正量也大，可能仍然收敛慢。后续可以考虑用 Heston 半解析解作为 Heston 的基线。

2. **v_max=1 的假设**：`sigma_eff = sqrt(v_n)` 中假设 `v_max=1`，所以 `v_n ≈ v`。如果 `v_max` 不是 1，需要在 `forward` 中传入 `v_max` 并做还原。当前代码中 `ModelParams.from_heston` 默认 `v_max=1.0`，所以没有问题。
