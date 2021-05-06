import setuptools

setuptools.setup(
    name="oglang",                                          # Replace with your own username
    version="0.0.1",
    author="NVIDIA",
    author_email="mmacklin@nvidia.com",
    description="High Performance Python DSL for Omniverse",
    long_description="",
    long_description_content_type="text/markdown",
                                                           #    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    package_data={"": ["native/*.h", "native/*.cpp", "native/*.cu", "kernels/core.dll"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy"],
)
