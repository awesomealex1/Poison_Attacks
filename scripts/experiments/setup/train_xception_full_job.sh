#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N train_xception_full       
#$ -cwd                  
#$ -l h_rt=48:00:00 
#$ -l h_vmem=80G
#$ -q gpu 
#$ -pe gpu-a100 1
. /etc/profile.d/modules.sh

cd /exports/eddie/scratch/s2017377/Poison_Attacks

# Load Python
module load anaconda
conda activate poison

# Run the program
. /exports/eddie/scratch/s2017377/Poison_Attacks/scripts/experiments/setup/train_xception_full.sh