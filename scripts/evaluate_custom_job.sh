#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N evaluate_custom         
#$ -cwd                  
#$ -l h_rt=16:00:00 
#$ -l h_vmem=40G
#$ -q gpu 
#$ -pe gpu-a100 1
. /etc/profile.d/modules.sh

cd /exports/eddie/scratch/s2017377/Poison_Attacks

# Load Python
module load anaconda
conda activate ff-gpu

# Run the program
. /exports/eddie/scratch/s2017377/Poison_Attacks/scripts/evaluate_custom.sh