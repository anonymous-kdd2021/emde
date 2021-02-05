import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="emde",
    version="0.0.1",
    description="Base EMDE package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    packages=setuptools.find_packages()
)
