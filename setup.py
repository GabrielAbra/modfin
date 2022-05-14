import setuptools
from modfin import __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='modfin',
    version=__version__,
    author='Gabriel Abrahao',
    author_email='gabrielabrahaorr@gmail.com',
    description='The modfin is a Python library, containing a set of tools useful  to perform quantitative analysis of financial assets.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/GabrielAbra/modfin',
    project_urls={
        "Bug Tracker": "https://github.com/GabrielAbra/modfin/issues",
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    license='MIT',
    install_requires=[
            "numpy",
            "pandas",
            "matplotlib",
            "scipy",
            "scikit-learn",
            "numba"],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",

)
