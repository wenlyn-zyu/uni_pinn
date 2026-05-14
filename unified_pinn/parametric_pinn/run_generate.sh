#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pinn_option
cd ~/zhuwl2022/united_pinn/parametric_pinn
mkdir -p results
python3 -u generate_ref_data.py --n-s 20 --n-tau 4 --n-k 5 --n-r 3 --n-v 4 --out results/ref_data.pkl 2>&1 | tee results/generate_ref.log
