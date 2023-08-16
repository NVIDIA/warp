import setuptools
import os
import shutil
import argparse

from typing import NamedTuple
from wheel.bdist_wheel import bdist_wheel

# Parse --build-option arguments meant for the bdist_wheel command. We have to parse these
# ourselves because when bdist_wheel runs it's too late to select a subset of libraries for package_data.
parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--platform", "-P", type=str, default="", help="Wheel platform: windows|linux|macos")
args = parser.parse_known_args()[0]


class Platform(NamedTuple):
    name: str
    fancy_name: str
    extension: str
    tag: str


platforms = [
    Platform("windows", "Windows", ".dll", "win_amd64"),
    Platform("linux", "Linux", ".so", "manylinux2014_x86_64"),
    Platform("macos", "macOS", ".dylib", "macosx_10_13_x86_64"),
]


# Determine supported platforms of warp/bin libraries based on their extension
def detect_warp_platforms():
    detected_platforms = set()
    for filename in os.listdir("warp/bin"):
        for p in platforms:
            if os.path.splitext(filename)[1] == p.extension:
                detected_platforms.add(p)

    if len(detected_platforms) == 0:
        raise Exception("No libraries found in warp/bin. Please run build_lib.py first.")

    return detected_platforms


detected_platforms = detect_warp_platforms()

wheel_platform = None  # The one platform for which we're building a wheel

if args.command == "bdist_wheel":
    if args.platform != "":
        for p in platforms:
            if args.platform == p.name or args.platform == p.fancy_name:
                wheel_platform = p
                print(f"Platform argument specified for building {p.fancy_name} wheel")
                break

        if wheel_platform is None:
            print(f"Platform argument '{args.platform}' not recognized")
        elif wheel_platform not in detected_platforms:
            print(f"No libraries found for {wheel_platform.fancy_name}")
            print(f"Falling back to auto-detection")
            wheel_platform = None

if wheel_platform is None:
    if len(detected_platforms) > 1:
        print("Libraries for multiple platforms were detected. Picking the first one.")
        print("Run `python -m build --wheel -C--build-option=-P[windows|linux|macos]` to select a specific one.")
    wheel_platform = next(iter(detected_platforms))

print("Creating Warp wheel for " + wheel_platform.fancy_name)


# Binary wheel distribution builds assume that the platform you're building on will be the platform
# of the package. This class overrides the platform tag.
# https://packaging.python.org/en/latest/specifications/platform-compatibility-tags
class WarpBDistWheel(bdist_wheel):
    # Even though we parse the platform argument ourselves, we need to declare it here as well so
    # setuptools.Command can validate the command line options.
    user_options = bdist_wheel.user_options + [
        ("platform=", "P", "Wheel platform: windows|linux|macos"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.platform = ""

    def get_tag(self):
        # The wheel's complete tag format is {python tag}-{abi tag}-{platform tag}.
        return "py3", "none", wheel_platform.tag

    def run(self):
        super().run()

        # Clean up so we can re-invoke `py -m build --wheel -C--build-option=--platform=...`
        # See https://github.com/pypa/setuptools/issues/1871 for details.
        shutil.rmtree("./build", ignore_errors=True)
        shutil.rmtree("./warp_lang.egg-info", ignore_errors=True)


# Distributions are identified as non-pure (i.e. containing non-Python code, or binaries) if the
# setuptools.setup() `ext_modules` parameter is not emtpy, but this assumes building extension
# modules from source through the Python build. This class provides an override for prebuilt binaries:
class BinaryDistribution(setuptools.Distribution):
    def has_ext_modules(self):
        return True


def get_warp_libraries(extension):
    libraries = []
    for filename in os.listdir("warp/bin"):
        if os.path.splitext(filename)[1] == extension:
            libraries.append("bin/" + filename)

    return libraries


if wheel_platform is not None:
    warp_binary_libraries = get_warp_libraries(wheel_platform.extension)
else:
    warp_binary_libraries = []  # Not needed during egg_info command

setuptools.setup(
    name="warp-lang",
    version="0.10.1",
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
        ]
        + warp_binary_libraries,
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
    cmdclass={
        "bdist_wheel": WarpBDistWheel,
    },
    install_requires=["numpy"],
    python_requires=">=3.7",
)
