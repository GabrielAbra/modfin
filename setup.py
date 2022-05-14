import setuptools

with open("modfin/__init__.py", "r") as version_file:
    VERSION = version_file.read()
    VERSION = VERSION.split("__version__ = ")[1][1:6]

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

NAME = "modfin"
AUTHOR = "Gabriel Abrahao"
AUTHOR_EMAIL = "gabrielabrahaorr@gmail.com"
DESCRIPTION = "Modules for Quantitative Financial Analysis"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = 'https://github.com/GabrielAbra/modfin'
REQUIRES_PYTHON = ">=3.8.0"

# Package requirements
INSTALL_REQUIRES = ['numpy', 'pandas', 'scipy',
                    'matplotlib', 'scikit-learn', 'numba']


setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=URL,

    project_urls={
        "Bug Tracker": "https://github.com/GabrielAbra/modfin/issues",
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Office/Business :: Financial :: Investment",
    ],

    license='MIT',
    install_requires=INSTALL_REQUIRES,
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)
