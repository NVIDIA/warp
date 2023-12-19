# This file is used to test reloading module references.

import warp as wp
import warp.tests.aux_test_reference_reference as refref

wp.init()


@wp.func
def magic():
    return 2.0 * refref.more_magic()
