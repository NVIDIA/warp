import setuptools

class BinaryDistribution(setuptools.Distribution):
    def has_ext_modules(foo):
        return True


setuptools.setup(
    name="warp",                                          
    version="0.1.25",
    author="NVIDIA",
    author_email="mmacklin@nvidia.com",
    description="A Python framework for high-performance simulation and graphics programming",
    long_description="",
    long_description_content_type="text/markdown",
    license="NVSCL",
    packages=setuptools.find_packages(),
    package_data={"": ["native/*.h", "native/*.cpp", "native/*.cu", "bin/*.dll", "bin/*.so", "gen/*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    distclass=BinaryDistribution,
    install_requires=["numpy"],
)
