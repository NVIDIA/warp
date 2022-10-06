import warp as wp
import warp.tests.test_dependency_dependency as depdep

wp.init()

@wp.func
def magic():
    return 2.0 * depdep.more_magic()
