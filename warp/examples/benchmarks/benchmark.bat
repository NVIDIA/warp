del benchmark.csv

python benchmark_cloth.py warp_cpu
python benchmark_cloth.py warp_gpu
@REM python benchmark_cloth.py taichi_cpu
@REM python benchmark_cloth.py taichi_gpu
python benchmark_cloth.py numpy
@REM python benchmark_cloth.py cupy
@REM python benchmark_cloth.py torch_cpu
@REM python benchmark_cloth.py torch_gpu
@REM python benchmark_cloth.py numba
@REM python benchmark_cloth.py jax_cpu
@REM python benchmark_cloth.py jax_gpu
@REM python benchmark_cloth.py paddle_cpu
@REM python benchmark_cloth.py paddle_gpu
