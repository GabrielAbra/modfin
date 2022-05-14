import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram

from modfin.portfolio_opt.base import portfolio_opt_base
from modfin.utils import riskmatrix_tools, portifolioopt_tools


class HierarchicalRiskParity(portfolio_opt_base):
    """
    Hierarchical Risk Parity Portfolio Optimization

    The Hierarchical Risk Parity algorithm,
    implements the allocation based on the book:

    `De Prado, Marcos Lopez. Advances in financial machine learning.
    John Wiley & Sons, 2018.` ISBN-10: 1119482089

    Obs.: The algorithm is specifically described in the Chapter 16 of the book.

    Parameters
    ----------

    RiskMatrix : `pd.DataFrame`
        A matrix of the risk implied by the returns of the assets.
        If possible from Modfin RiskMatrix Module.

    Method : `str`, (optional)
        Method of linkage used in the Hierarchical Clustering.
        Available methods are:
            single (default) , ward, average, complete, median, weighted, centroid

        For more information about the linkage methods, check the scipy documentation.
            Scipy : https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

    Functions
    ---------

    optimize() : Calculate the optimal portfolio allocation using the Hierarchical Risk Parity algorithm.

    plot : Plot the algorithm decision tree.
    """

    def __init__(self, RiskMatrix: pd.DataFrame, Method: str = 'Single'):

        # Define the method of linkage to be used
        if Method.lower() not in [
                "ward",
                "single",
                "average",
                "complete",
                "median",
                "weighted",
                "centroid"]:

            raise ValueError(
                "Method must be one of the following: 'Ward', 'Single', 'Average', 'Complete', 'Median', 'Weighted' or 'Centroid'")  # noqa
        self._Method = Method.lower()

        # Check if the RiskMatrix is valid.
        self._RiskMatrix = self._check_rm(RiskMatrix)

        # Get Asset names
        self._asset_names = self._get_asset_names(RiskMatrix)

        # Get the number of assets
        self._num_assets = len(self._asset_names)

        # Define the start clusters as empty
        self.clusters = None

    def _getQuasiDiag(cluster, num_assets, curr_index):
        # Sort clustered items by distance
        if curr_index < num_assets:
            return [curr_index]
        left = int(cluster[curr_index - num_assets, 0])
        right = int(cluster[curr_index - num_assets, 1])
        return (HierarchicalRiskParity._getQuasiDiag(cluster, num_assets, left) + HierarchicalRiskParity._getQuasiDiag(cluster, num_assets, right))

    def _getClusterVar(covariance: pd.DataFrame, cluster_indices):
        # Calculate the variance of the clusters
        cluster_covariance = covariance.iloc[cluster_indices, cluster_indices]
        parity_w = HierarchicalRiskParity._getIVP(cluster_covariance)
        cluster_variance = portifolioopt_tools.expected_variance(
            covariance_matrix=cluster_covariance, weights=parity_w)
        return cluster_variance

    def _getIVP(cov: pd.DataFrame):
        ivp = 1 / np.diag(cov.values)
        ivp = ivp * (1 / np.sum(ivp))
        return ivp

    def _getRecBipart(covariance_matrix, assets, ordered_indices):
        # Compute HRP allocation
        weights = pd.Series(1, index=ordered_indices)

        # Initialize all clusters with the same alpha score
        clustered_alphas = [ordered_indices]

        while clustered_alphas:
            # bi-section of cluster
            clustered_alphas = [cluster[start:end]
                                for cluster in clustered_alphas
                                for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                                if len(cluster) > 1]

            for subcluster in range(0, len(clustered_alphas), 2):
                left_cluster = clustered_alphas[subcluster]
                right_cluster = clustered_alphas[subcluster + 1]
                left_cluster_variance = HierarchicalRiskParity._getClusterVar(
                    covariance_matrix, left_cluster)
                right_cluster_variance = HierarchicalRiskParity._getClusterVar(
                    covariance_matrix, right_cluster)
                alloc_factor = 1 - left_cluster_variance / \
                    (left_cluster_variance + right_cluster_variance)

                # Apply the allocation factor to the left and right cluster
                weights[left_cluster] *= alloc_factor
                weights[right_cluster] *= 1 - alloc_factor

        # Apply the resulting weights to the assets
        weights.index = assets[ordered_indices]
        return weights

    def optimize(self) -> pd.DataFrame:
        """
        Calculate the optimal portfolio allocation using the Hierarchical Risk Parity algorithm.

        Parameters
        ----------
        AssetPrices : py:class:`pandas.DataFrame` with the daily asset prices

        Returns
        -------
        Portifolio : `pandas.DataFrame` with the weights of the portfolio
        """

        # Calculate the correlation matrix from the risk matrix
        corr_matrix = riskmatrix_tools.cov_to_corr(self._RiskMatrix)

        # Calculate the euclidian distance between all assets correlations
        distance_matrix = np.sqrt((1 - corr_matrix).round(10) / 2)

        # Tree clustering the distance matrix
        clusters = scipy_linkage(squareform(
            distance_matrix), method=self._Method)
        self.clusters = clusters

        # Quasi diagnalization of the tree clusters
        ordered_indices = HierarchicalRiskParity._getQuasiDiag(
            clusters, self._num_assets, (2 * self._num_assets - 2))

        # Recursive Bisection from ordered asset indices
        Weight = HierarchicalRiskParity._getRecBipart(
            covariance_matrix=self._RiskMatrix, assets=self._asset_names, ordered_indices=ordered_indices)

        # Create and return the portfolio
        return self._pandas_portifolio(Weight, self._asset_names)

    def plot(self, Size=6):
        """
        Plot the algorithm decision tree.

        Parameters
        ----------
        Size : `int`, Size of the figure. The default is 6.

        """
        # Check if clusters is defined
        if self.clusters is None:
            raise ValueError(
                "You must run the Optimize method before plotting")

        # Plot the dendrogram
        fig = plt.figure(num=None, figsize=(1.25 * Size, Size), dpi=80)
        ax1 = fig.add_subplot(111)
        ax1.title.set_text('Hierarchical Risk Parity Dendrogram')
        ax1 = dendrogram(self.clusters, labels=self._asset_names,
                         show_leaf_counts=True)
        plt.grid(False)
        plt.tight_layout()
        plt.show()
