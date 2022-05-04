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
The ModFin project aims to provide users with the necessary tools for modeling and analyzing individual assets and portfolios. This package contains various modules that provide a variety of useful functions and algorithms. Here is a table of the implemented modules:

- [Asset Screening](#asset-screening)
- [Portfolio Optimization](#portfolio-optimization)
- [Risk Matrix](#risk-matrix)

*Note: Modules to be fully implemented on live libraries.*

- *Analysis of Time Series*
- *Bet Sizing*
- *Data Structures*
- *Online Portfolio*
- *Option Pricing*

&nbsp;
<!---- install ----->
# Installation

### Requirements

ModFin requires **Python 3.8 or later** and **C++ build tools**.

If needed, you can install the C++ build tools on [Visual Studio](https://visualstudio.microsoft.com/downloads/).


### Methods
The project is available on PyPI, and can be installed with [pip](https://pip.pypa.io/en/stable/installing/#install-command-requirements-file) package manager with the following command:

```
$ pip install modfin
```

Alternatively, install the package from the source using the following command:

```
git clone https://github.com/GabrielAbra/modfin
python setup.py install
```

&nbsp;
<!---- modules ----->

# Asset Screening
The `AssetScreening` module offers functions for screening assets based on a predetermined set of metrics. One fundamental approach is to screen assets based on smart betas (Style factors), when a given a combination of traits, are likely to be of interest to a specific investor. The ready to use functions groups are:

### Metrics
This submodule provides bundles of functions that can be used to screen assets. Some of the metrics are:

* #### Risk Metrics
    * Beta, Downside Beta, Beta Quotient
    * RSquaredScore
    * LPM
    * ...
* #### Return Metrics
    * Annualized Return
    * Exponencial Returns
    * Log Returns
    * ...
* #### Ratio Metrics
    * Omega Ratio
    * Sortino Ratio
    * Tail Ratio
    * ...

### Screening
This submodule provides functions that can be used to screen assets. Some of the screening functions are:

* #### Z-Score Screening
* #### Sequential Screening
* #### Quantile Screening
&nbsp;

# Risk Matrix
The `RiskMatrix` module provides multiple functions for analyzing time series data and generating risk matrices. There are three different types of algorithms that can be distinguished as:

* ### Sample
    * Covariance
    * Semicovariance
* ### Estimator
    * Empirical Covariance
    * Minimum Covariance Determinant
* ### Shrinkage
    * Shrinkage (Basic Shrinkage)
    * LedoitWolf (Ledoit-Wolf Shrinkage Method)
    * Oracle (Oracle Approximating Shrinkage)

&nbsp;

# Portfolio Optimization
The `PortfolioOpt` module provides algorithms for optimization of an a asset portfolios. The algorithms live implemented algorithms are:

* ### Risk Parity
    The `RiskParity` algorithm is a simple algorithm that optimizes a portfolio based on the risk parity weighting.
* ### Hierarchical Risk Parity
    The `HierarchicalRiskParity` (HRP) algorithm, implements the allocation based on the book: 
    > De Prado, Marcos Lopez. Advances in financial machine learning. John Wiley & Sons, 2018.

    The algorithm is a risk based optimisation, which has been shown to generate diversified portfolios with robust out-of-sample properties.
* ### Inverse Variance
    The `InverseVariance` algorithm is a simple algorithm that optimizes a portfolio based on the provided risk matrix (usually the covariance matrix).

* ### Efficient Frontier
    The `EfficientFrontier` algorithm provide a portfolio allocation based on the modern portfolio theory (MPT). The algoritms was first proposed by Harry Markowitz in the paper:

    >Markowitz, H.M.. Portfolio Selection. The Journal of Finance, 1952.
    
&nbsp;
<!---- license ----->

# License

ModFin is released under the MIT license, so the code is open source and can be used in any project, provided that the original author is credited.




