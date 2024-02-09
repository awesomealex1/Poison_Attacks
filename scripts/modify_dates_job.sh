
#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N modify_dates              
#$ -cwd                  
#$ -l h_rt=16:00:00 
#$ -l h_vmem=10G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem
# Initialise the environment modules
. /etc/profile.d/modules.sh

cd /exports/eddie/scratch/s2017377/Poison_Attacks

# Load Python
module load anaconda
conda activate ff-gpu

# Run the program
. /exports/eddie/scratch/s2017377/Poison_Attacks/scripts/modify_dates.sh