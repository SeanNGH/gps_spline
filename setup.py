from setuptools import setup, find_packages

setup(
    name="gpsspline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy", "scipy", "matplotlib", "pyproj", "pandas"
    ],
)
