import warp as wp

# CPU-mode reproducer adapted from your CUDA snippet


def main():
    # Try CUDA first, fall back to CPU
    try:
        device = wp.get_device("cuda:0")
    except Exception:
        device = wp.get_device("cpu")

    @wp.kernel
    def touch(x: wp.array(dtype=wp.float32)):
        i = wp.tid()
        if i < x.shape[0]:
            x[i] = x[i] + 1.0

    # Load the module after kernels are registered. Only swallow load failures
    # for CPU devices; on GPU we want to propagate real errors.
    try:
        wp.load_module(device=device)
    except Exception:
        if not device.is_cpu:
            raise

    # CPU devices do not have CUDA streams; run a CPU-friendly capture path.
    if device.is_cpu:
        wp.capture_begin(device=device, force_module_load=False)
        try:
            for _ in range(4):
                t = wp.empty(4096, dtype=wp.float32, device=device)
                wp.launch(touch, dim=4096, inputs=[t])
                del t
        finally:
            g = wp.capture_end(device=device)

        for _ in range(8):
            wp.capture_launch(g)
    else:
        main_stream = wp.get_stream(device)
        sub_stream = wp.Stream(device)

        wp.capture_begin(device=device, stream=main_stream, force_module_load=False)
        try:
            sub_stream.wait_stream(main_stream)
            with wp.ScopedStream(sub_stream, sync_enter=False):
                for _ in range(4):
                    t = wp.empty(4096, dtype=wp.float32, device=device)
                    wp.launch(touch, dim=4096, inputs=[t], stream=sub_stream)
                    del t
            main_stream.wait_stream(sub_stream)
        finally:
            g = wp.capture_end(device=device, stream=main_stream)

        replay = wp.Stream(device)
        for _ in range(8):
            wp.capture_launch(g, stream=replay)
        wp.synchronize_stream(replay)


if __name__ == "__main__":
    try:
        main()
        print("Reproducer finished successfully")
    except Exception as e:
        print("Reproducer failed:", e)
        raise
