#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pinn_option
cd ~/zhuwl2022/united_pinn/parametric_pinn
mkdir -p results
echo "=== Training started at $(date) ==="
echo "GPU: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))")"
echo ""
python3 -u train_parametric.py \
  --ref results/ref_data_fast.pkl \
  --epochs 50000 \
  --lr 1e-3 \
  --hidden 256 \
  --depth 8 \
  --out results/fully_param_v1.pt \
  --device cuda \
  --save-every 5000 \
  --w-data 100.0 \
  --w-pde 1.0 \
  --w-bc 10.0 \
  --w-ic 10.0 \
  --w-bsm-raw 1.0 \
  2>&1
echo ""
echo "=== Training finished at $(date) ==="
