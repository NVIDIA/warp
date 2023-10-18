import os

import coverage

cover = coverage.Coverage(
    source=["warp", "warp.sim", "warp.render"],
    omit=[
        "build*.py",
        "copyright.py",
        "setup.py",
        os.path.join("warp", "stubs.py"),
        os.path.join("warp", "tests", "*.py"),
        os.path.join("warp", "thirdparty", "appdirs.py"),
        os.path.join("warp", "fem", "**", "*.py"),
        os.path.join("warp", "render", "render_opengl.py"),
    ],
)

cover.config.disable_warnings = [
    "module-not-measured",
    "module-not-imported",
    "no-data-collected",
    "couldnt-parse",
]

cover.exclude("@wp")
cover.exclude("@warp")
cover.start()

import test_all  # noqa: E402

test_all.run()


cover.stop()
cover.save()

cover.html_report()
