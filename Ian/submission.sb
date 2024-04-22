#!/bin/bash
########## Define Resources Needed with SBATCH Lines ##########
#SBATCH --time=00:02:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1           # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem=1G                    # memory required per node - amount of memory (in bytes)
#SBATCH --job-name=CUDA_p1     # you can give your job a name for easier identification (same as -J) 
#SBATCH --gpus=v100:1               # I implicitly trust other people
 
########## Command Lines to Run ##########

module purge
module load gcc/7.3.0-2.30 OpenMPI HDF5
module load NVHPC

srun ./a.out