#!/bin/bash
#SBATCH --job-name=Tranformer_inelasticity_beta_loss
#SBATCH --partition=gr10_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

#SBATCH --gres=gpu:1
#SBATCH --mem=31000
#SBATCH --time=00:30:00
#SBATCH --output=SLURM_logs/output/job_output_%j.txt
#SBATCH --error=SLURM_logs/error/job_error_%j.txt

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/moust/miniconda3/lib/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/moust/miniconda3/envs/icet2/lib/

cd /groups/icecube/moust/work/IceCubeEncoderTransformer

source /groups/icecube/moust/miniconda3/envs/graphnet/bin/activate icet2

srun python3 src/train.py experiment=kaggle_2nd_place_inelasticity_SLURM