#!/bin/bash -l
### Job Name
#PBS -N Monsoon_S2S
### Charging account
#PBS -A P66770001
### Request one chunk of resources with 1 CPU and 10 GB of memory
#PBS -l select=1:ncpus=1:mem=10GB
### Allow job to run up to 40 minutes
#PBS -l walltime=02:00:00
### Route the job to the casper queue
#PBS -q casper
### Join output and error streams into single file
#PBS -j oe


# to start this scrip run "sbatch sbatch.sh $mo $ba"
# to check process:  qstat -u $USER

mo=4
ba=3

echo $mo 
echo $ba
echo '----------'

PythonName="NMME_WTing.py"
echo $PythonName

# module load python/3.7.5
ncar_pylib
# ml ncl nco
./$PythonName $mo $ba


exit
