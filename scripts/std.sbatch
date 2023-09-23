#!/bin/bash

#SBATCH --job-name=std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --priority=4294967293
#SBATCH --account=class
#SBATCH --out=./out/std.out


module purge;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

singularity exec --nv \
  --overlay /scratch/$USER/my_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python main.py --batch-size=64 --lr=0.00025 --discount-factor=0.99 --num-epochs=50 --memory-size=10000 --frame-repeat=12 --steps-per-epoch=2000 --epsilon-decay=0.9996 --model='dqn' --checkpoint='std'"
