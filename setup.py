from setuptools import setup
import strain_tools

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="strain_tools",
    version=strain_tools.__version__,
    author="Tom Chudley",
    author_email="chudley.1@osu.edu",
    url="https://github.com/trchudley/glacier-strain-tools",
    py_modules=["strain_tools"],
    description="Tools for deriving surface-parallel strain rates from glacier velocity fields.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["numpy", "rasterio", "numba",],
    license="MIT",
    # python_requires=">=3.7.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
)
