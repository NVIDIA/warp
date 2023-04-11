import warp as wp


@wp.kernel
def compute():

    v = wp.transform_identity()

    f = wp.curlnoise())