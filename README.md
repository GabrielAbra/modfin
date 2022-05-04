# ModFin
<!---- shields ----->

<p align="center">
    <a href="https://pypi.org/project/modfin/">
        <img src="https://img.shields.io/pypi/pyversions/modfin"
            alt="python version"></a> &nbsp;
    <a href="https://pypi.org/project/modfin/">
        <img src="https://img.shields.io/pypi/v/modfin"
            alt="pypi version"></a> &nbsp;
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"
            alt="MIT license"></a> &nbsp;
</p>

<!---- Desc ----->
The ModFin project aims to provide users with the necessary tools for modeling and analyzing individual assets and portfolios. This package contains various modules that provide a variety of useful functions and algorithms. Here is a list of the modules:
- [Analysis of Financial Time Series](#analysis-of-financial-time-series)
- [Asset Screening](asset-screening)
- [Bet Sizing](#Bet-Sizing)
- [Data Structures](#Data-Structures)
- [Filtering](#Filtering)
- [Online Portfolio](#Online-Portfolio)
- [Option Pricing](#Option-Pricing)
- [Portfolio Optimization](#Portfolio-Optimization)

*Note: Not all modules are fully implemented on live libraries.*


# Installation

## Requirements

ModFin requires **Python 3.8 or later** and **C++ build tools**.

If needed, you can install the C++ build tools on [Visual Studio](https://visualstudio.microsoft.com/downloads/).


## Methods
The project is available on PyPI, and can be installed with [pip](https://pip.pypa.io/en/stable/installing/#install-command-requirements-file) package manager with the following command:

```
$ pip install modfin
```

Alternatively, install the package from the source using the following command:

```
git clone https://github.com/GabrielAbra/modfin
python setup.py install
```


# Analysis of Financial Time Series

## Risk Matrix
The Risk Matrix Module provides multiple functions for analyzing time series data and generating risk matrices. There are three different types of algorithms that can be distinguished as `Sample`, `Estimator` and `Shrinkage` algorithms.

### Sample Algorithms

- Covariance
- SemiCovariance

### Estimator Algorithms

- Minimum Covariance Determinant 
- Empirical Covariance

### Shrinkage Algorithms

- Shrinkage (Basic Shrinkage)
- LedoitWolf (Ledoit-Wolf Shrinkage Method)
- Oracle (Oracle Approximating Shrinkage)





# Asset Screening

. . .

. .

.

# Portfolio Optimization

. . .

. .

.

# License

ModFin is released under the MIT license, so the code is open source and can be used in any project, provided that the original author is credited.




