## Project 5 Final Report

Please note that the final code is called Project5_Diffusion_Code.cu.

This is run using `nvcc Project5_Diffusion_Code.cu -DDEBUG -o diffusion` and the sbatch submission file `sbatch project5_slurm.sb`. 

You need to also run `module load NVHPC/21.9-GCCcore-10.3.0-CUDA-11.4`

Any of the slurm files are located in folder called Anna.


1. Report your timings for the host, naive CUDA kernel, shared memory CUDA kernel,
and the excessive memory copying case, using block dimensions of 256, 512,
and 1024. Use a grid size of `2^15+2*NG` (or larger) and run for 100 steps (or
shorter, if it's taking too long). Remember to use `-O3`!

Please note that timing is in ms/step, the code was run for 100 steps and const unsigned int n = (1<<15) +2*NG;

|      | Host    | Naïve      | Shared     | Excessive |
| ---- | ------- | ---------- | ---------- | --------- |
| 256  | 0.26258 | 0.00351584 | 0.00288032 | 0.0946173 |
| 512  | 0.24843 | 0.00337024 | 0.00288384 | 0.0955744 |
| 1024 | 0.24682 | 0.00352832 | 0.0030272  | 0.0942624 |

Please note that timing is in ms/step, the code was run for 100 steps and const unsigned int n = (1<<11) +2*NG;

|      | Host      | Naïve      | Shared     | Excessive |
| ---- | --------- | ---------- | ---------- | --------- |
| 256  | 0.0154209 | 0.00270336 | 0.00251904 | 0.020728  |
| 512  | 0.0155497 | 0.002816   | 0.00265216 | 0.0197814 |
| 1024 | 0.0158    | 0.0029184  | 0.00277504 | 0.0204307 |

2. How do the GPU implementations compare to the single threaded host code. Is it
faster than the theoretical performance of the host if we used all the cores on
the CPU?

The naïve and shared CUDA implementations are both about 75 times faster than the CPU Host implementation.

**add more**

3. For the naive kernel, the shared memory kernel, and the excessive `memcpy` case,
which is the slowest? Why? How might you design a larger code to avoid this slow down?

The excessive `memcpy` case is the slowest. **add more**

4. Do you see a slow down when you increase the block dimension? Why? Consider
that multiple blocks may run on a single multiprocessor simultaneously, sharing
the same shared memory.

We do see a slow down as the block dimension increases. **add more**
