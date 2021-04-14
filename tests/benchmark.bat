rm report.csv
python benchmark_cloth.py oglang_cpu
python benchmark_cloth.py oglang_gpu
python benchmark_cloth.py taichi_cpu
python benchmark_cloth.py taichi_gpu
REM python benchmark_cloth.py numpy
REM python benchmark_cloth.py cupy
REM python benchmark_cloth.py torch_cpu
REM python benchmark_cloth.py torch_gpu
REM python benchmark_cloth.py numba
