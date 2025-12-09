# Instructions

## Setup
This project was developed and executed on lxplus9, along with its associated GPU node. First clone this repository
```
ssh lxplus9.cern.ch
git clone https://github.com/max-zhao0/tac-hep-gpus.git
cd tac-hep-gpus/Project
```
Install vtune.
```
python3.12 -m venv .venv
source .venv/bin/activate
pip install dpcpp-cpp-rt
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/a04c89ad-d663-4f70-bd3d-bb44f5c16d57/intel-vtune-2025.7.0.248_offline.sh
sh intel-vtune-2025.7.0.248_offline.sh
source ~/intel/oneapi/setup_vars.sh
```
## CPU
Compile the example written in pure single-threaded C++. This script prints out the matrices at each step, and performs direct validation listed in shared.hpp.
```
g++ cpp_part.cpp -o cpp.out -I. -g
```
This will yield a time of roughly:
```
time ./cpp.out
> real    0m0.316s
> user    0m0.310s
> sys     0m0.005s
```
Now profile the performance with vtune.
```
vtune -collect hotspots -result-dir r001hs -- ./cpp.out
vtune -report hotspots -r r001hs -format=csv -csv-delimiter=semicolon >cpp.hotspots.csv
```
You will notice that the matrix multiplication is significantly more expensive than the stenciling, which is to be expected since it is cubic. The result can be found in cpp.hotspots.csv in the repository.

## GPU
### Setup
These examples will run on the lxplus9 GPU nodes
```
ssh lxplus-gpu.cern.ch
cd tac-hep-gpu/Project
```
Note that the first half of this project was carried on the non-GPU node because I could not get vtune to run properly on the GPU node.

### Basic
First we compile a basic cuda version of the example.
```
nvcc basic_cuda_part.cu -o basic_cuda.out -I.
```
This example is essentially a direct port of the cpp example to cuda. The same printing and validation are being run.
```
time ./basic_cuda.out
> real    0m0.474s
> user    0m0.104s
> sys     0m0.365s
```
It should be noted that the CPU example, when compiled and run on the GPU node runs about twice as slow. Now profile with nsys.
```
nsys profile -o basic_cuda --stats=true ./basic_cuda.out > basic_cuda_profile.txt
```
It is evident that a significant amount of the runtime is dominated by memory copying.

### Managed memory
Now compile the managed memory example, which is the same as the basic except with host pointers replaced with cuda managed memory.
```
nvcc mm_cuda_part.cu -o mm_cuda.out -I.
```
The timing of the program is
```
time ./mm_cuda.out
> real    0m0.475s
> user    0m0.102s
> sys     0m0.368s
```
Profiling with nsys again, we see that the device to host is significantly faster, but using the paged memory has significantly slowed down the stenciling kernel.
```
nsys profile -o mm_cuda --stats=true ./mm_cuda.out > mm_cuda_profile.txt
```

### Optimized
Now compiled the optimized example.
```
nvcc optimized_cuda_part.cu -o optimized_cuda.out -I.
```
This example splits into two non-default streams, which copy memory and run the stenciling kernel on A and B in parallel. One stream performs the matrix multiplication while the other copies one of the stenciled matrices back to host for printing. Both kernels now also used shared memory to form a cache. The stenciling kernel uses the same caching strategy as in the week4 homework. The matrix multiplication kernel caches the section of A used by the thread block. Note that this is on the edge of the shared memory possible, and increasing DSIZE from 512 is likely to break this.
```
time ./optimized_cuda.out
> real    0m0.455s
> user    0m0.074s
> sys     0m0.373s
```
A marginal improvement is observed.
```
nsys profile -o optimized_cuda --stats=true ./optimized_cuda.out > optimized_cuda_profile.txt
```

### Alpaka
No Alpaka version of these programs was successfully compiled for this project, even for the minimal basic_cuda_part.cu version. 
