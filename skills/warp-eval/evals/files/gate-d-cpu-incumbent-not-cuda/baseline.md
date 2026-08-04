# Query baseline

The optional `native_index` backend runs optimized C++ on the CPU. It does not
execute CUDA kernels or use an NVIDIA GPU.

On the supplied x86-64 host it processes 10,000 `query_first` inputs in 2.5 ms.
Production also calls `query(mode="all", return_metadata=True)` heavily; its
frequency and distribution are not represented by the supplied maintenance
profile.

An optional NVIDIA backend is permitted, subject to the normal authorization
checkpoint.
