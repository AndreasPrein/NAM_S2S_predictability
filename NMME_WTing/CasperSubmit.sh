#!/bin/bash -l
# to start this scrip run "./CasperSubmit.sh"
# to check the status run "squeue -u $USER"

for mo in {0..4};
do
    for ba in {0..3};
    do
        sed "s/MOD/$mo/g" sbatch.sh > sbatch_fin.sh
        sed -i "s/BAS/$ba/g" sbatch_fin.sh
        qsub sbatch_fin.sh
    done
done


exit
