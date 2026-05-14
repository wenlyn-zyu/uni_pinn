#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pinn_option
cd ~/zhuwl2022/united_pinn/parametric_pinn
mkdir -p results
echo "Starting fast ref data generation at $(date)"
python3 -u generate_ref_fast.py --n-bsm 200 --n-cev 300 --n-heston 400 --n-s 30 --n-v 4 --out results/ref_data_fast.pkl 2>&1
echo "Finished at $(date)"
