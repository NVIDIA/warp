# Contact compaction CUDA baseline

`contact_cuda.compact_contacts` is the production implementation. The CUDA team
owns it, tests it on every supported GPU, and has maintained its Python and C++
ABI for four releases. It preserves candidate order, reports the actual count
and overflow, and passes the full degeneracy and capacity suite.

The release benchmark uses CUDA-resident inputs with 16 million candidate pairs
on an H100 SXM:

| Measurement | Result |
|---|---:|
| End-to-end stage latency | 0.82 ms |
| Effective device-memory bandwidth | 2.7 TB/s |
| Measured copy/scan roofline for this representation | 3.1 TB/s |
| Peak temporary memory | 64 MiB |

The implementation therefore reaches 87% of the measured bandwidth ceiling.
Its latency and memory already satisfy the production contract. The request has
no maintainability, ergonomics, packaging, autodiff, or new-functionality
objective; performance is the only objective under consideration.
