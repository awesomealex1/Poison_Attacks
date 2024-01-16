cd /exports/eddie/scratch/s2017377/Poison_Attacks
touch goldenfile
find /exports/eddie/scratch/s2017377/Poison_Attacks -type f -exec touch -r goldenfile {} \;
rm goldenfile