#!/bin/bash

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 1 CPU core
#SBATCH -n 1

#SBATCH -t 00:05:00
#SBATCH -o with_cpu.out
#SBATCH -e with_cpu.err

module load cuda/12.2.2  gcc/10.2

make clean all
./my_program > out.ppm
python3 ./ppmtojpg.py