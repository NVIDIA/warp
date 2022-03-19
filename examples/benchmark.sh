rm outputs/benchmark.csv

python3 benchmark_cloth.py warp_cpu
python3 benchmark_cloth.py warp_gpu
# python3 benchmark_cloth.py taichi_cpu
# python3 benchmark_cloth.py taichi_gpu
# python3 benchmark_cloth.py numpy
# python3 benchmark_cloth.py cupy
# python3 benchmark_cloth.py torch_cpu
# python3 benchmark_cloth.py torch_gpu
# python3 benchmark_cloth.py jax_cpu
# python3 benchmark_cloth.py jax_gpu
# python3 benchmark_cloth.py numba
