#!/bin/bash -l

#SBATCH --job-name=test_KIN
#SBATCH --output=./zSlurm_output/%x_%j.out
#SBATCH --mail-user="nieuwhof@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

#SBATCH --partition="testing"
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G

echo "#### Starting test"

cd ..

conda activate nbody

module load OpenMPI/5.0.3-GCC-13.3.0

python test_KIN/debug.py 
# python test_KIN/test_KIN.py --param_file test_KIN/test_params2.json --job_id $SLURM_JOB_ID

echo "#### Finished Test. Have a nice day"