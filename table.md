| Predicate | Workload | Native (µs) | `warp.geometry` (µs) | Speedup |
| --- | --- | ---: | ---: | ---: |
| tri–tri | random | 31.044 | 14.545 | 2.13× |
| tri–tri | coplanar | 32.298 | 17.003 | 1.90× |
| tri–tri (float64) | random | 209.252 | 177.381 | 1.18× |
| seg–tri vs. ray–tri Möller | random | 11.260 | 11.258 | 1.00× |
| aabb–aabb | random | 9.157 | 9.137 | 1.00× |
| seg–aabb vs. robust ray | random | 9.327 | 9.143 | 1.02× |
| tri–aabb vs. cuBQL | random | 19.793 | 11.410 | 1.73× |
| tri–aabb vs. cuBQL | contained | 11.329 | 11.329 | 1.00× |
