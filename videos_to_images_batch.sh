#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N videos_to_images              
#$ -cwd                  
#$ -l h_rt=00:16:00 
# -l h_vmem=40G
# -q gpu 
# -pe gpu-a100 1
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem
# Initialise the environment modules
. /etc/profile.d/modules.sh

cd /exports/eddie/scratch/s2017377/Poison_Attacks/scripts/

# Load Python
module load anaconda
conda activate ff

# Run the program
. /exports/eddie/scratch/s2017377/Poison_Attacks/scripts/videos_to_images_ff.sh