#!/bin/sh
#SBATCH -J xxx
#SBATCH -p 36C_dell
#SBATCH -N 1
#SBATCH -c 20
#SBATCH -o ../log/t20.out
#SBATCH -e ../log/t20.err
#SBATCH -D /public3/home/yangsheng

module load anaconda/3
ulimit -s unlimited
export OMP_NUM_THREADS=1

for((i=0;i<5;i++))
do
"D:\anaconda3\envs\TNtest\python.exe"  simuplus.py
done
