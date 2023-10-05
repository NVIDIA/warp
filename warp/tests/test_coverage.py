import coverage
import os

cover = coverage.Coverage(source=os.path.join(os.path.dirname(__file__), "test_misc.py"),
                          omit=["build*.py",
                                "copyright.py",
                                "setup.py",
                                os.path.join("warp", "stubs.py"),
                                os.path.join("warp", "tests", "*.py"),
                                os.path.join("warp", "fem", "*.py"),
                                os.path.join("warp", "fem", "field", "*.py"),
                                os.path.join("warp", "fem", "geometry", "*.py"),
                                os.path.join("warp", "fem", "quadrature", "*.py"),
                                os.path.join("warp", "fem", "space", "*.py")])

cover.config.disable_warnings = [
    "module-not-measured",
    "module-not-imported",
    "no-data-collected",
    "couldnt-parse",
]

cover.exclude("@wp")
cover.exclude("@warp")
cover.start()

import test_all
test_all.run()

cover.stop()
cover.save()

cover.html_report()