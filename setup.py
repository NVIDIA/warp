import setuptools

setuptools.setup(
    name="warp",                                          
    version="0.1.24",
    author="NVIDIA",
    author_email="mmacklin@nvidia.com",
    description="High Performance Python DSL for Omniverse",
    long_description="",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={"": ["native/*.h", "native/*.cpp", "native/*.cu", "kernels/*.dll", "kernels/*.so"]},
    classifiers=[
        "Prwpramming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy"],
)
