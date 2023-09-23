#!/bin/bash

#SBATCH --job-name=oduel_rocket
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --priority=4294967293
#SBATCH --account=class
#SBATCH --out=./out/oduel_rocket.out

module purge;

singularity exec --nv \
  --overlay /scratch/$USER/my_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python main.py --batch-size=128 --lr=0.007124271448356194 --discount-factor=0.8828099164309334 --num-epochs=50 --memory-size=10000 --frame-repeat=12 --steps-per-epoch=2000 --epsilon-decay=0.9956338864091042 --model='oduel_dqn' --checkpoint='oduel_rocket' --scenario='rocket_basic'"
