def register(parent):
    # Only register the test class if CUDA is available
    if wp.is_cuda_available():
        parent.add_class(TestStratifiedOptim)

if __name__ == "__main__":
    wp.init()
    unittest.main()
