#!/usr/bin/env bash
#SBATCH -p emergency_gpu
#SBATCH --gres=gpu:1
#SBATCH --chdir=/hpc2hdd/home/cwu319/RC/RPG_KDD2025
#SBATCH -o /hpc2hdd/home/cwu319/RC/RPG_KDD2025/history/%x-%j.out
#SBATCH -e /hpc2hdd/home/cwu319/RC/RPG_KDD2025/history/%x-%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=rpg-baseline

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/hpc2hdd/home/cwu319/RC/RPG_KDD2025}"
CATEGORY="${1:-Sports_and_Outdoors}"

case "$CATEGORY" in
  Sports_and_Outdoors)
    LR=0.003
    TEMPERATURE=0.03
    N_CODEBOOK=16
    NUM_BEAMS=100
    N_EDGES=30
    PROPAGATION_STEPS=5
    ;;
  Beauty)
    LR=0.01
    TEMPERATURE=0.03
    N_CODEBOOK=32
    NUM_BEAMS=20
    N_EDGES=200
    PROPAGATION_STEPS=3
    ;;
  Toys_and_Games)
    LR=0.003
    TEMPERATURE=0.03
    N_CODEBOOK=16
    NUM_BEAMS=200
    N_EDGES=20
    PROPAGATION_STEPS=3
    ;;
  CDs_and_Vinyl)
    LR=0.001
    TEMPERATURE=0.03
    N_CODEBOOK=64
    NUM_BEAMS=20
    N_EDGES=500
    PROPAGATION_STEPS=5
    ;;
  *)
    echo "Unsupported category: $CATEGORY" >&2
    exit 2
    ;;
esac

RUN_ID="${RUN_ID:-baseline_${CATEGORY}_st5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO_ROOT"

source /hpc2hdd/home/cwu319/anaconda3/etc/profile.d/conda.sh
conda activate rpg

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export USE_TF="${USE_TF:-0}"

echo "repo_root=$REPO_ROOT"
echo "category=$CATEGORY"
echo "run_id=$RUN_ID"
echo "host=$(hostname)"
echo "started_at=$(date '+%F %T')"
nvidia-smi || true

python main.py \
  --category="$CATEGORY" \
  --run_id="$RUN_ID" \
  --sent_emb_model=sentence-transformers/sentence-t5-base \
  --sent_emb_dim=768 \
  --sent_emb_pca=128 \
  --lr="$LR" \
  --temperature="$TEMPERATURE" \
  --n_codebook="$N_CODEBOOK" \
  --num_beams="$NUM_BEAMS" \
  --n_edges="$N_EDGES" \
  --propagation_steps="$PROPAGATION_STEPS" \
  $EXTRA_ARGS

echo "finished_at=$(date '+%F %T')"
