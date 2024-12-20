#!/bin/bash
#SBATCH --partition=gpu          # Use high-performance GPU partition if available
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1   # Use 1 GPU
#SBATCH --time=08:00:00          # Increase training time
#SBATCH --job-name=gpu_run
#SBATCH --mem=64GB               # Allocate sufficient memory
#SBATCH --cpus-per-task=8        # Increase CPU resources
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

# Ensure that Conda is initialized in the current environment
eval "$(conda shell.bash hook)"

# Activate the virtual environment
conda activate flappy_env_3

export PYTHONPATH=/home/parekh.meh/5100_final_project:$PYTHONPATH

# Navigate to the directory containing your script
cd /home/parekh.meh/5100_final_project/flappy_bird_gym

# Load optimized libraries (optional)
module load intel-mkl
module load cuda/11.8

# Execute your Python script
python flappy_dqn_rgb.py
