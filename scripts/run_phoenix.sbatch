#!/bin/bash
#SBATCH -J meta_mole_ttt               # Job name
#SBATCH -N 1                           # Number of nodes
#SBATCH -n 1                           # Number of tasks
#SBATCH -c 8                           # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Request 1 GPU (e.g. A100 / V100 / RTX 6000)
#SBATCH --mem=48G                      # Memory allocation
#SBATCH -t 12:00:00                    # Wall time limit (HH:MM:SS)
#SBATCH -q inferno                     # Queue (inferno on GT PACE / Phoenix)
#SBATCH -o logs/slurm_%j.out           # Standard output log
#SBATCH -e logs/slurm_%j.err           # Standard error log

# -------------------------------------------------------------
# Georgia Tech PACE Phoenix Cluster SLURM Job Script
# Meta-MoLE-TTT: Two-Stage Meta-Learning with Orthogonal Scratchpad
# -------------------------------------------------------------

echo "=========================================================="
echo "Starting job on host: $(hostname)"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "=========================================================="

# Create log directory
mkdir -p logs artifacts checkpoints

# Load Phoenix Modules (adjust module names if using specific CUDA versions)
module purge
module load anaconda3
module load cuda/12.1

# Activate conda environment (create with: conda create -n mole python=3.10)
# source activate mole
# or:
# conda activate mole

# Check GPU availability
nvidia-smi

# 1. Run Unit Tests to ensure environment integrity
echo ">>> Running Unit Tests..."
python -m pytest tests/

# 2. Run Pipeline (Example with Tier 0 or Tier 1 config)
# Change config to configs/tier1_qwen05b_ablation.yaml or configs/tier2_llama8b_main.yaml as needed
CONFIG="configs/tier1_qwen05b_ablation.yaml"

echo ">>> Running Stage 1 ERM..."
python run_pipeline.py --config $CONFIG --stage 1

echo ">>> Running Gate Expansion Surgery..."
python run_pipeline.py --config $CONFIG --stage surgery

echo ">>> Running Stage 2 FOMAML Meta-Training..."
python run_pipeline.py --config $CONFIG --stage 2

echo ">>> Running Evaluation (5-Row Ablation)..."
python run_pipeline.py --config $CONFIG --stage eval

echo ">>> Generating Analysis Visualizations..."
python run_pipeline.py --config $CONFIG --stage plot

echo "=========================================================="
echo "Job finished at: $(date)"
echo "=========================================================="
