#!/bin/bash -l

#SBATCH --job-name=run_KIN_v
#SBATCH --output=./zSlurm_output/%x_%j.out
#SBATCH --mail-user="nieuwhof@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-long"
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=5G

#SBATCH --constraint=ib

echo "#### Starting test"

cd ..

conda activate nbody

python Validation/validation.py --param_file Validation/params_job1.json
echo "Processing results..."
python Validation/process_results.py --param_file Validation/params_job1.json --job_id $SLURM_JOB_ID  

echo "#### Finished Test. Have a nice day"