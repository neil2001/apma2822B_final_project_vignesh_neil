#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p 3090-gcondo --gres=gpu:1 --gres-flag=enforce-binding
#SBATCH --mem=40G
# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 1 CPU core
#SBATCH -n 1

#SBATCH -t 00:15:00
#SBATCH -o with_gpu.out
#SBATCH -e with_gpu.err

# Load CUDA module
module load cuda/12.2.2  gcc/10.2   


nvidia-smi

# Compile CUDA program and run
#nvcc -arch sm_20 vecadd.cu -o vecadd
# nvcc -O2 src/main.cu
make
# ./cuda_program
./cuda_program > out.ppm
# nsys profile --stats=true --force-overwrite=true --output=outputs/gpu_report ./a.out > out.ppm
# ./a.out > out.ppm

SECONDS=0
# Your function or command here
python3 ./ppmtojpg.py
# End timing
duration=$SECONDS
echo "Python file took $duration seconds"

echo "done"

