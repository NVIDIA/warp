del outputs\benchmark.csv

python benchmark_cloth.py warp_cpu
python benchmark_cloth.py warp_gpu
python benchmark_cloth.py taichi_cpu
python benchmark_cloth.py taichi_gpu
python benchmark_cloth.py numpy
python benchmark_cloth.py cupy
python benchmark_cloth.py torch_cpu
python benchmark_cloth.py torch_gpu
python benchmark_cloth.py numba
python benchmark_cloth.py jax_cpu
python benchmark_cloth.py jax_gpu


