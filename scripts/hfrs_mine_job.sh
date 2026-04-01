#!/usr/bin/env bash
#SBATCH -p emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH --chdir=/hpc2hdd/home/cwu319/RC/method
#SBATCH -o /hpc2hdd/home/cwu319/RC/method/history/hfrs-mine-%j.out
#SBATCH -e /hpc2hdd/home/cwu319/RC/method/history/hfrs-mine-%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=hfrs-mine

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/hpc2hdd/home/cwu319/RC/method}"
BASE_CKPT="${BASE_CKPT:?BASE_CKPT is required}"
HFRS_OUTPUT="${HFRS_OUTPUT:?HFRS_OUTPUT is required}"
CATEGORY="${CATEGORY:-Beauty}"
SENT_EMB_MODEL="${SENT_EMB_MODEL:-/hpc2hdd/home/cwu319/RC/method/st5base}"
SENT_EMB_DIM="${SENT_EMB_DIM:-768}"
SENT_EMB_PCA="${SENT_EMB_PCA:-128}"
N_CODEBOOK="${N_CODEBOOK:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TOPM="${TOPM:-128}"

cd "$REPO_ROOT"

source /hpc2hdd/home/cwu319/anaconda3/etc/profile.d/conda.sh
conda activate rpg

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

echo "repo_root=$REPO_ROOT"
echo "category=$CATEGORY"
echo "checkpoint=$BASE_CKPT"
echo "output=$HFRS_OUTPUT"
echo "host=$(hostname)"
echo "started_at=$(date '+%F %T')"
nvidia-smi || true

python scripts/mine_hfrs_pool.py \
  --checkpoint="$BASE_CKPT" \
  --category="$CATEGORY" \
  --sent_emb_model="$SENT_EMB_MODEL" \
  --sent_emb_dim="$SENT_EMB_DIM" \
  --sent_emb_pca="$SENT_EMB_PCA" \
  --n_codebook="$N_CODEBOOK" \
  --batch_size="$BATCH_SIZE" \
  --topm="$TOPM" \
  --output="$HFRS_OUTPUT"

echo "finished_at=$(date '+%F %T')"
