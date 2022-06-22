import setuptools

setuptools.setup(
    name="probpy",
    version="0.0.1",
    author="Jonas Valfridsson",
    author_email="jonas@valfridsson.net",
    description="Probabilistic utilities for numpy",
    url="https://github.com/JonasRSV/probpy",
    packages=setuptools.find_packages(),
    install_requires=["numpy==1.22.0", "numba==0.48"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
