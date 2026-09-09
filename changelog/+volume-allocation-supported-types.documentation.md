Drop the incorrect CUDA-only restriction from `wp.Volume.allocate()` and `wp.Volume.load_from_numpy()`, which
work on CPU devices too, and correct the list of value types their `bg_value` argument accepts.
