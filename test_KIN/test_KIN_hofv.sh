#!/bin/bash -l

echo "Starting KIN test..."

cd ..

conda activate nbody

#python test_KIN/debug.py
nohup python test_KIN/test_KIN.py --param_file test_KIN/test_params.json --job_id test_hofvijver > test_KIN/zSlurm_output/test_hofvijver.txt &

echo "Finished learning."
