import setuptools
import os
from wheel.bdist_wheel import bdist_wheel


def get_warp_platform():
    for filename in os.listdir("warp/bin"):
        if os.path.splitext(filename)[1] == ".dll":
            return "win_amd64"
        if os.path.splitext(filename)[1] == ".so":
            return "manylinux2014_x86_64"
        if os.path.splitext(filename)[1] == ".dylib":
            return "macosx_10_13_x86_64"

    raise Exception("No libraries found in warp/bin")


# Distributions are identified as non-pure (i.e. containing non-Python code, or binaries) if the
# setuptools.setup() `ext_modules` parameter is not emtpy, but this assumes building extension
# modules from source through the Python build. This class provides an override for prebuilt binaries:
class BinaryDistribution(setuptools.Distribution):
    def has_ext_modules(self):
        return True


# Binary wheel distribution builds assume that the platform you're building on will be the platform
# of the package. This factory function provides a class which overrides the platform tag.
# https://packaging.python.org/en/latest/specifications/platform-compatibility-tags
def bdist_factory(platform_tag: str):
    class warp_bdist(bdist_wheel):
        def get_tag(self, *args, **kws):
            # The tag format is {python tag}-{abi tag}-{platform tag}.
            return "py3", "none", platform_tag

    return warp_bdist


setuptools.setup(
    name="warp-lang",
    version="0.9.0",
    author="NVIDIA",
    author_email="mmacklin@nvidia.com",
    description="A Python framework for high-performance simulation and graphics programming",
    url="https://github.com/NVIDIA/warp",
    project_urls={
        "Documentation": "https://nvidia.github.io/warp",
    },
    long_description="",
    long_description_content_type="text/markdown",
    license="NVSCL",
    packages=setuptools.find_packages(),
    package_data={
        "": [
            "native/*.cpp",
            "native/*.cu",
            "native/*.h",
            "native/clang/*.cpp",
            "native/nanovdb/*.h",
            "tests/assets/*",
            "bin/*",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": bdist_factory(get_warp_platform())},
    install_requires=["numpy"],
    python_requires=">=3.7",
)
