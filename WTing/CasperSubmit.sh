#!/bin/bash -l
#SBATCH -J RF
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH -t 24:00:00
#SBATCH -A P66770001
#SBATCH -p dav

# to start this scrip run "for ii in {0..7}; do echo $ii; sbatch CasperSubmit.sh $ii HUC2_XWTs_search_full; done"
# to chack the status run "squeue -u $USER"

HUCnr="$1"
SetupFile="$2"

echo $HUCnr
echo $SetupFile

#module load python/2.7.14
module load python/3.7.5
ncar_pylib
ml mkl
ml ncl nco
# source my private Python copy on Casper
source /glade/work/prein/PYTHON_CASPER_clone/bin/activate

#srun ./SearchOptimum_XWT-Combination.py $HUCnr
srun ./SearchOptimum_XWT-Combination.py $HUCnr $SetupFile

exit
