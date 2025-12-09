g++ cpp_part.cpp -o cpp.out -I. -g
real    0m0.316s
user    0m0.310s
sys     0m0.005s

vtune -collect hotspots -result-dir r001hs -- ./cpp.out
vtune -report hotspots -r r001hs -format=csv -csv-delimiter=semicolon >cpp.hotspots.csv

nvcc basic_cuda_part.cu -o basic_cuda.out -I.
real    0m0.474s
user    0m0.104s
sys     0m0.365s

nsys profile -o basic_cuda --stats=true ./basic_cuda.out > basic_cuda_profile.txt

nvcc mm_cuda_part.cu -o mm_cuda.out -I.
real    0m0.475s
user    0m0.102s
sys     0m0.368s

nsys profile -o mm_cuda --stats=true ./mm_cuda.out > mm_cuda_profile.txt

nvcc optimized_cuda_part.cu -o optimized_cuda.out -I.
real    0m0.455s
user    0m0.074s
sys     0m0.373s

nsys profile -o optimized_cuda --stats=true ./optimized_cuda.out > optimized_cuda_profile.txt
