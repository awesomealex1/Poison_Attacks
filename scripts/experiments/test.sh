#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N xception_full_transfer         
#$ -cwd                  
#$ -l h_rt=1:00:00 
#$ -l h_vmem=40G
. /etc/profile.d/modules.sh

cd /exports/eddie/scratch/s2017377/Poison_Attacks

# Load Python
module load anaconda
conda activate ff-gpu

# Run the program
python /exports/eddie/scratch/s2017377/Poison_Attacks/test.py