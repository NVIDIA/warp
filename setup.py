import setuptools

setuptools.setup(
    name="warp-lang",
    version="0.3.1",
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
    package_data={"": ["native/*", "native/nanovdb/*", "tests/assets/*", "bin/*", "gen/*"]},
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy"],
    python_requires=">=3.7"
)