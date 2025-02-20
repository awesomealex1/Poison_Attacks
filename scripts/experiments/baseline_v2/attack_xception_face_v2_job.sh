#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N attack_xception_face_baseline_v2
#$ -cwd                  
#$ -l h_rt=48:00:00 
#$ -l h_vmem=1000G
#$ -q gpu 
#$ -pe gpu-a100 1
. /etc/profile.d/modules.sh

cd /exports/eddie/scratch/s2017377/Poison_Attacks

# Load Python
module load anaconda
conda activate poison

# Run the program
. /exports/eddie/scratch/s2017377/Poison_Attacks/scripts/experiments/baseline_v2/attack_xception_face_v2.sh