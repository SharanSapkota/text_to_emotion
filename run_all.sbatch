#!/bin/bash
# Submit with:  mkdir -p logs && sbatch run_all.sbatch
# Lines marked TODO must be adjusted to Roihu (check the CSC docs / `module avail pytorch`).
#SBATCH --job-name=goemotions
#SBATCH --account=project_XXXXXXX      # TODO: your CSC project
#SBATCH --partition=gpu                # TODO: Roihu GPU partition name
#SBATCH --gres=gpu:1                   # TODO: GPU request syntax on Roihu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -e
module purge
module load pytorch                    # TODO: exact module name on Roihu
# If you made a venv (python -m venv --system-site-packages venv), activate it:
[ -f venv/bin/activate ] && source venv/bin/activate

# Compute nodes usually have no internet: use only what prepare_data.py cached.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "== Baseline: TF-IDF + Logistic Regression =="
python baseline_lr.py --n_jobs $SLURM_CPUS_PER_TASK

echo "== BERT fine-tuning =="
python train_bert.py --model_path models/bert-base-uncased --epochs 3 --num_workers 4

echo "== Report assets =="
python make_report_assets.py
