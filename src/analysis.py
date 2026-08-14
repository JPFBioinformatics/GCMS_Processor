"""
Methods for statistical analysis of the data matrices (typically use normalized)

Preprocessing
-------------

    - Drop sparse features
    - Missing value imputation (half-minimum or KNN)
    - Log2 transformation
    - Autoscaling (mean-center + unit variance)

    Preprocessing order is:
        keep = find_sparse_features(dm.missing, dm.group_indices)
        matrix = dm.data['norm_area'][:,keep].copy()
        matrix = impute_nans(matrix, method="hm")
        matrix = log_transform(matrix)
        matrix = autoscale(matrix)
        analyze matrix ...
        NOTE - be sure to create a COPY of the matrix from dm.data, not to use a view, or you will
        alter the data in dm permanantly with processing

Univariate
----------

    - Welch's t-test and Mann-Whitney U for two-group comparisons
    - One-way ANOVA and Kruskal-Wallis for multi-group comparisons
    - Benjamini-Hochberg FDR correction
    - Log2 fold change

Multivariate
------------

    - PCA (unsupervised dimensionality reduction)
    - PLS-DA (supervised group discrimination)
    - Permutation testing for PLS-DA model validation
    - Hierarchical clustering
    - Pearson/Spearman correlation matrix
    - Correlation Matrix (spearman-monotonic or pearson-linear)

Feature Selection
-----------------

    - VIP scores from PLS-DA
    - Significant feature filtering by p-value and fold change threshold
    - Most correlated features

All functions take numpy arrays as input and return numpy arrays or dicts.
Group indices are lists of row indices into the sample dimension of the matrix.

Basic use guide
---------------

# preprocess
keep = find_sparse_features(dm.missing, dm.group_indices)
matrix = dm.data['norm_area'][:, keep].copy()               NOTE - MUST COPY MATRIX HERE, DO NOT USE VIEW
matrix = impute_nans(matrix)
matrix = log2_transform(matrix)
matrix = autoscale(matrix)

# decide PCA components
scree = scree_data(matrix)
# plot scree['cumulative'] vs component number, pick n_for_80pct

# two-group comparison
p_vals = t_test(matrix, group_a, group_b)
p_corr = fdr_correction(p_vals)
fc = log2_fold_change(matrix, group_a, group_b)
d = effect_size(matrix, group_a, group_b)
sig = significant_features(p_corr, fc)
# volcano plot: fc on x, -log10(p_corr) on y, color by d

# PLS-DA with validation
pls = pls_da(matrix, labels)
perm = permutation_test(matrix, labels)   # is it better than chance?
cv = pls_da_cv(matrix, labels)            # how well does it predict?
vips = vip_scores(pls['model'])           # which features drive separation?

"""

# region Imports

import numpy as np

from sklearn.experimental import enable_iterative_imputer   # needed to enable IterativeImputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.ensemble import RandomForestRegressor
from scikit_posthocs import posthoc_dunn

from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, kruskal, spearmanr, pearsonr

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from src.config_loader import ConfigLoader


# logging
import logging
logger = logging.getLogger(__name__)

# endregion

# region                 ---------- Preprocessing ----------

def find_sparse_features(missing: np.ndarray, group_indices: dict, threshold: int = 0.5):
    """
    Drops features if ALL groups have >threshold missingness 

    Params
    ------
        missing                     bool missing matrix from DataMatrix
        group_indices               dict of grp_name: list of indices
        threshold                   missingness threshold to drop feature
    
    Returns
    -------
        keep                        list of feature indices to keep
    """
    keep = []
    for i in range(missing.shape[1]):
        for _, indices in group_indices.items():
            grp_missing = missing[indices, i]
            if np.mean(grp_missing) <= threshold:
                keep.append(i)
                break
    
    return keep

def impute_nans(matrix: np.ndarray, method: str = "hm", neighbors: int = 5):
    """
    imputes nan values of non-sparse columns using method sepcified, half-minimum is sufficient for
    most metaboloimcs applications
    DOES NOT copy matrix, edits in place - do that yourself befor using

    Params
    ------
        matrix                      matrix we are imputing
        method                      "hm" or "knn" based on what imputation method to use
        neighbors                   number of neighbors for knn imputation
    """

    if method not in ["hm", "knn","mice","rf"]:
        logger.warning(
            "Imputation method must be" \
            "\nhm   =   Half-Minimum" \
            "\nknn  =   K-nearest neighbor" \
            "\nmice =   Multiple Imputation by Chained Equations" \
            "\nrf   =   Random Forest" \
            "\nDefaulting to half-minimum")
        method = "hm"

    # half minimum
    if method == "hm":
        for j in range(matrix.shape[1]):
            col = matrix[:,j]
            min_val = np.nanmin(col)
            col[np.isnan(col)] = min_val / 2
            matrix[:,j] = col

    # k-nearest neightbor
    elif method == "knn":
        imputer = KNNImputer(n_neighbors = neighbors)
        matrix = imputer.fit_transform(matrix)

    # multiple imputation by chained equations
    elif method == "mice":
        imputer = IterativeImputer(max_iter=10, random_state=0)
        matrix = imputer.fit_transform(matrix)

    # random forest
    elif method == "rf":
        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100), max_iter=10, random_state=0)
        matrix = imputer.fit_transform(matrix)

    return matrix

def log2_transform(matrix: np.ndarray):
    """
    log2 transforms a given matrix
    DOES NOT copy matrix, edits in place

    Params
    ------
        matrix                      data matrix with sparse features dropped and nan imputed

    Returns
    -------
        log2 transformed matrix
    """
    return np.log2(matrix + 1e-10) # 1e-10 to prevent log(0)

def autoscale(matrix: np.ndarray):
    """
    autoscales matrix, making each feature have mean 0 and std 1 to prevent high abundance metaoblits
    from dominating pca
    DOES NOT copy matrix, edits in place

    Params
    ------
        matrix                      data matrix withs parse features dropped, nan imputed, and log2 transformed

    Returns
    -------
        autoscaled matrix
    """
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    stds[stds == 0] = 1                     # prevent divide by 0 for constant columns
    return (matrix - means) / stds

def full_preprocess(matrix: np.ndarray, missing: np.ndarray, group_indices: dict, cfg: ConfigLoader):
    """
    fully preproccesses a matrix, finds and drops sparse features, imputes nan values, log2 transforms data, then
    autoscales distributions

    Params
    ------
    matrix                      matrix to process
    missing                     missingness matrix corresponding to matrix
    group_indices               dict of group_name: list of index values in matrix for samples in that group (rows)
    cfg                         configloader object for this run

    Returns
    -------
    mat                         preprocessed matrix
    keep                        array of indices to keep
    """

    # get requried data
    sparseness_threshold = cfg.get("sparseness_threshold")
    if not sparseness_threshold:
        logger.warning("No sparseness threshold detected, defaulting to 0.5")
        sparseness_threshold = 0.5

    imputation_method = cfg.get("impute_method")
    if not imputation_method:
        logger.warning("No imputation method detected, defaulting to hm")
        imputation_method = 'hm'
        
    impute_knns = cfg.get("impute_knns")
    if not impute_knns:
        logger.warning("No nearest neighbors setting detected, defaulting to 5")
        impute_knns = 5
    
    # copy input matrix
    mat = matrix.copy()

    # figure out which features to drop and drop
    keep = find_sparse_features(missing, group_indices, sparseness_threshold)
    mat = mat[:,keep]

    # impute remaining nan values
    mat = impute_nans(mat, imputation_method, impute_knns)

    # log2 transform to correct typical right-skewing of metabolomic distributions
    mat = log2_transform(mat)

    # autoscale to remove dominance from high abundance features
    mat = autoscale(mat)

    return mat, keep

# endregion

# region                 ---------- Univariate ----------

def t_test(matrix: np.ndarray, group_a: list, group_b: list):
    """
    Performs a Welch's t-test per feature between two groups where you CAN assume approximate normality

    Params
    ------
        matrix                      matrix of values to test
        group_a/_b                  indices of each group

    Returns
    -------
        p_values                    array of p-values matched to feature indices
    """
    p_values = np.zeros(matrix.shape[1])
    for j in range(matrix.shape[1]):
        a = matrix[group_a,j]
        b = matrix[group_b,j]
        _,p = ttest_ind(a, b, equal_var=False, nan_policy='omit')
        p_values[j] = p
    return p_values

def mann_whitney(matrix: np.ndarray, group_a: list, group_b: list,):
    """
    Performs mann-whitney non-parametric U test between two groups when you CANNOT assume approximate normality

    Params
    ------
        matrix                      matrix of values to test
        group_a/_b                  indices of each group

    Returns
    -------
        p_values                    array of p_values matched to feature indices
    """
    p_values = np.zeros(matrix.shape[1])
    for j in range(matrix.shape[1]):
        a = matrix[group_a,j]
        b = matrix[group_b,j]
        _,p = mannwhitneyu(a, b, alternative='two-sided')
        p_values[j] = p
    return p_values

def effect_size(matrix: np.ndarray, group_a: list, group_b: list):
    """
    Calculates chron's d effect size per feature between two groups, measures magnitude of difference independent
    of samples size; d < 0.2 small, d 0.2 - 0.8 medium, d > 0.8 large

    Params
    ------
        matrix                      matrix for analysis (samples, features)
        group_a/_b                  row indices for groups

    Returns
    -------
        array (n_features) of chron's D values index matched to features in matrix
    """
    a = matrix[group_a, :]
    b = matrix[group_b, :]

    mean_a = np.nanmean(a,axis=0)
    mean_b = np.nanmean(b,axis=0)

    std_a = np.nanstd(a,axis=0,ddof=1)
    std_b = np.nanstd(b,axis=0,ddof=1)

    n_a = len(group_a)
    n_b = len(group_b)

    pooled_std = np.sqrt(((n_a-1) * std_a**2 + (n_b-1) * std_b**2) / (n_a + n_b - 2))
    pooled_std[pooled_std == 0] = np.nan    # prevents dividing by 0

    return (mean_a - mean_b) / pooled_std

def ow_anova(matrix: np.ndarray, group_indices: dict, posthoc: bool = True):
    """
    Performs one-way anova, parametric extension of t-test to 3+ groups, tells you that some group
    differs, not which group - pair with post-hoc tests to determine which group differs
    Assumes groups are normally distributed

    Params
    ------
        matrix                      matrix of values to test (samples x features)
        group_indices               dict of group: list of indices for each group (dm.group_indices)
        posthoc                     wether or not to run turkey HSD posthoc analysis

    Returns
    -------
        result{
            pvals: array of p values from one way anova
            posthoc_pvals: 3D array (n_features, n_groups, n_groups) for all group comparisons from turkey HSD
            group_names: list of group_names for matching to n_groups indices in posthoc_pvals
        }
    """
    p_values = np.zeros(matrix.shape[1])
    for j in range(matrix.shape[1]):
        groups = [matrix[idxs,j] for idxs in group_indices.values()]
        _,p = f_oneway(*groups)
        p_values[j] = p
    
    result = {'p_values': p_values}

    if posthoc:
        posthoc_pvals, group_names = _turkey_posthoc(matrix, group_indices)
        result['posthoc_pvals'] = posthoc_pvals
        result['group_names'] = group_names

    return result

def _turkey_posthoc(matrix: np.ndarray, group_indices: dict):
    """
    runs Turkey's HSD parametric test following ANOVA analysis, use IF there are significant
    results found in ANOVA to find out which groups are contributing to this difference

    Params
    ------
        matrix                      matrix to analyze
        group_indices               indices of each group in the sample set

    Returns
    -------
        group_names                 list of group names, index matched to pvals 3D array
        pvals                       3D array (n_features, n_groups, n_groups), pvals[j] gives full p-value matrix
                                    (n_groups, n_groups) for feature j, pvals[:,i,k] gives you the pvalue for 
                                    group i vs group k across all features at once, pvals[j,i,k] gives you a single
                                    p-value for one feature j between groups i and k

    """
    n_groups = len(group_indices)
    n_features = matrix.shape[1]
    pvals = np.full((n_features,n_groups,n_groups), np.nan)
    group_names = list(group_indices.keys())

    for j in range(matrix.shape[1]):
        values = []
        labels = []
        for group_name, indices in group_indices.items():
            values.extend(matrix[indices,j])
            labels.extend([group_name] * len(indices))

        turkey = pairwise_tukeyhsd(values,labels)
        
        for row in turkey.summary().data[1:]:
            g1,g2,_,p,_,_,_ = row
            i = group_names.index(g1)
            k = group_names.index(g2)
            pvals[j,i,k] = p
            pvals[j,k,i] = p

    return pvals, group_names

def kruskal_wallis(matrix: np.ndarray, group_indices: dict, posthoc: bool = True):
    """
    Performs Kruskal-Wallis test, non-parametric extension of mann-whitney to 3+ groups with same
    relationship to one-way anova as mann-whitney has to t-test - still need post-hoc tests
    DOES NOT assume groups are normally distributed

    Params
    ------
        matrix                      matrix of values to test (samples x features)
        group_indices               dict of group: list of indices for each group (dm.group_indices)
        posthoc                     wether or not to run posthoc analysis

    Returns
    -------
        result{
            pvals: array of p values from one way anova
            posthoc_pvals: 3D array (n_features, n_groups, n_groups) for all group comparisons from dunn posthoc
            group_names: list of group_names for matching to n_groups indices in posthoc_pvals
        }
    """
    p_values = np.zeros(matrix.shape[1])
    for j in range(matrix.shape[1]):
        groups = [matrix[idxs,j] for idxs in group_indices.values()]
        try:
            _,p = kruskal(*groups)
        except:
            p = np.nan
        p_values[j] = p

    result = {'p_values': p_values}

    if posthoc:
        posthoc_pvals, group_names = _dunn_posthoc(matrix, group_indices)
        result['posthoc_pvals'] = posthoc_pvals
        result['group_names'] = group_names

    return result

def _dunn_posthoc(matrix: np.ndarray, group_indices: dict):
    """
    Runs pairwise Dunn test, used IF kruskall wallis returns signficiant resutls to determine
    which groups are contributing to this significance

    Params
    ------
        matrix                      matrix to analyze
        group_indices               dict of group indices lists

    Returns
    -------
        group_names                 list of group names, index matched to pvals 3D array
        pvals                       3D array (n_features, n_groups, n_groups), pvals[j] gives full p-value matrix
                                    (n_groups, n_groups) for feature j, pvals[:,i,k] gives you the pvalue for 
                                    group i vs group k across all features at once, pvals[j,i,k] gives you a single
                                    p-value for one feature j between groups i and k
    """
    n_groups = len(group_indices)
    n_features = matrix.shape[1]
    pvals = np.full((n_features,n_groups,n_groups), np.nan)
    group_names = list(group_indices.keys())

    for j in range(n_features):
        data = [matrix[indices,j] for indices in group_indices.values()]
        result = posthoc_dunn(data, p_adjust='bh')
        pvals[j] = result.values

    return pvals, group_names

def fdr_correction(p_values: np.ndarray, method: str = 'bh'):
    """
    Performs fdr correction on a set of p values to reduce fasle positives.  Default 'bh' is Benhamini-Hotchburg
    which is good for discovery metabolomics where false positives are preferable to missing real hits, could
    also do Bonferroni ('bonferroni') for a more strict correction

    Params
    ------
        p_values                    array of p_values for correction
        method                      how to perform correction, 'bh' for discovery 'bonferronni' for more strict
    
    Returns
    -------
        corrected                   FDR corrected array
    """
    _, corrected, _, _ = multipletests(p_values, method=method)
    return corrected

def log2_fold_change(matrix: np.ndarray, group_a: list, group_b: list):
    """
    Calculates log2 fold change between groups per feature

    Params
    ------
        matrix                      matrix of values to test
        group_a/_b                  indices of each group

    Returns
    -------
        log2 fold change per feature array
    """
    mean_a = np.nanmean(matrix[group_a,:], axis=0)
    mean_b = np.nanmean(matrix[group_b,:], axis=0)
    return np.log2(mean_a / (mean_b + 1e-10))       # 1e-10 to prevent divide by 0

# endregion

# region                 ---------- Multivariate ----------

def pca(matrix: np.ndarray, exp_var: float = 0.8, n_components=None):
    """
    Performs PCA, unsuperviesd dimensionality reduction to inspect overall data structure

    Params
    ------
        matrix                      matrix of values to test
        exp_var                     amount of variance you wnat to explain with PCs (just gives idx of pc where this value is surpassed)
        n_components                number of components to calculate, None means do max
        
    Returns
    -------
        {
        scores: (n_samples,n_components) - position of samples in reduced space for plotting
        loadings: (n_components,n_features) - loading arrays showing which features drive seperation
        explained_variance: (n_components) - fraciton of total varaince each PC explains
        }
    """
    if exp_var > 1 or exp_var < 0:
        logger.warning("Explained Varaince must be in range [0,1]")
        exp_var = 0.80

    model = PCA(n_components=n_components)
    scores = model.fit_transform(matrix)

    cumulative = np.cumsum(model.explained_variance_ratio_)
    n_for_exp_var = int(np.searchsorted(cumulative, exp_var)) + 1

    return {
        'scores': scores,
        'loadings': model.components_,
        'explained_variance': model.explained_variance_ratio_,
        'cumulative_variance': cumulative,
        'n_for_exp_var': n_for_exp_var
    }

def scree_data(matrix: np.ndarray, exp_var: float):
    """
    Returns varianace explianed per component for skerr plot, used to decide how many components capture
    meaningful variance

    Params
    ------
        matrix                      matrix (n_samples, n_features) for analysis

    Returns
    -------
        {
        explained: array (n_components) varaince explained per component
        cumulative: array (n_components) cumulative varaince explained
        n_for_80pct: scalar, minimum components needed to explain 80% varaince
        }
    """
    if exp_var > 1:
        logger.warning("Explained Varaince must be in range [0,1]")
        return
    
    n_components = min(matrix.shape[0], matrix.shape[1])
    model = PCA(n_components=n_components)
    model.fit(matrix)

    explained = model.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n_for_exp_var = int(np.searchsorted(cumulative, exp_var)) + 1

    return {
        'explained': explained,
        'cumulative': cumulative,
        'n_for_exp_var': n_for_exp_var
    }

def pls_da(matrix: np.ndarray, labels: list, n_components: int = 2):
    """
    Perfroms PLS-DA, supervised vesrion of PCA that maximizes group sepeartion but risks overfitting,
    permutation testing is required to determine significance

    Params
    ------
        matrix                      matrix of valuest to test
        labels                      list of labels index matched to samples
        n_components                number of dimenstions to extract

    Returns
    -------
        {
        scores:(n_samples,n_components) - position of samples in reduced space for plotting
        loadings: (n_components,n_features) - loading arrays showing which features drive seperation
        model: sklearn fitted model object to use later for vip_scores()
        }
    """
    max_comps = min(matrix.shape[0], matrix.shape[1])
    if n_components > max_comps:
        logger.warning(f'Too many PCA components requested, defautling to max {max_comps}')
        n_components = max_comps

    y = LabelEncoder().fit_transform(labels)
    model = PLSRegression(n_components=n_components)
    model.fit(matrix, y)
    scores = model.transform(matrix)
    return {
        'scores': scores,
        'loadings': model.x_loadings_,
        'model': model
    }

def pls_da_cv(matrix: np.ndarray, labels: list, n_components: int = 2, cv: str = 'loo') -> dict:
    """
    Cross-validated PLS-DA. More rigorous than permutation test for small n, LOO holds out one sample at 
    a time and predicts its group.

    Params
    ------
        matrix                      samples x features matrix
        labels                      group label per sample, index matched
        n_components                number of PLS components
        cv                          'loo' for leave-one-out, 'kfold' for 5-fold

    Returns
    -------
        {
        accuracy: fraction of samples correctly predicted
        predictions: (n_samples) predicted labels for each held-out sample
        confusion: (n_groups, n_groups) confusion matrix
        }
    """
    y = LabelEncoder().fit_transform(labels)
    predictions = np.zeros(len(y), dtype=int)

    cv_splitter = LeaveOneOut() if cv == 'loo' else KFold(n_splits=5, shuffle=True)

    for train_idx, test_idx in cv_splitter.split(matrix):
        model = PLSRegression(n_components=n_components)
        model.fit(matrix[train_idx], y[train_idx])
        pred = model.predict(matrix[test_idx])
        predictions[test_idx] = np.round(pred).astype(int).clip(0, len(set(y)) - 1).ravel()

    encoder = LabelEncoder().fit(labels)
    return {
        'accuracy': accuracy_score(y, predictions),
        'predictions': encoder.inverse_transform(predictions),
        'confusion': confusion_matrix(y, predictions)
    }

def permutation_test(matrix: np.ndarray, labels: list, n_perms: int = 999):
    """
    Validates PLS-DA by checking if group separation is better than chance, shuffles group labels and 
    rebuilds model each time, used to determine if the real model is actually significantly better than a 
    random assortment of label assignments (combats overfitting of pls-da)

    Params
    ------
        matrix                      matrix of values to test
        labels                      index matched list of sample labels
        n_perms                     maximum number of permutations of label shuffling

    Returns
    -------
        {
        real_r2: scalar - R squared value of the actual model with true group labels
        perm_r2s: (n_perms) - array of R squared values from each permutated model
        p_value: scalar - fraction of permutations that beat true model (<0.05 means fewer than 5% of random perms beat true model)
        }
    """
    y = LabelEncoder().fit_transform(labels)
    model = PLSRegression(n_components=2)
    model.fit(matrix, y)
    real_r2 = r2_score(y, model.predict(matrix))
    perm_r2s = []
    for _ in range(n_perms):
        y_perm = np.random.permutation(y)
        m = PLSRegression(n_components=2)
        m.fit(matrix, y_perm)
        perm_r2s.append(r2_score(y_perm, m.predict(matrix)))
    p_value = np.mean(np.array(perm_r2s) >= real_r2)
    return {
        'real_r2': real_r2,
        'perm_r2s': np.array(perm_r2s),
        'p_value': p_value
        }

def hierarchical_clustering(matrix: np.ndarray, method: str = 'ward'):
    """
    Hierarchical clustering for heatmap ordering.
    Performs hirearchical clustering on data to group similar samples together for heatmap visualization,
    ward linkage minimizes within-cluster variance, optimal leaf ordering reorders leaves to minimize
    adjacent dissimilartiy

    Params
    ------
        matrix                      matrix of values to test
        method                      which method to use for seperation

    Returns
    -------
        {
        row_linkage: linkage matrix for samples - encodes full clustering tree, passed to scipy dendrogram/seaborn clustermap
        col_linkage: same as row_linkage for features
        row_order: (n_samples) - optimal row (sample) order indices for heatmap, reaorder matrix rows by matrix[row_order,:]
        col_order: (n_features) - same as row_order but for columns/features
        }
    """
    row_linkage = linkage(matrix, method=method)
    col_linkage = linkage(matrix.T, method=method)
    row_order = leaves_list(optimal_leaf_ordering(row_linkage, matrix))
    col_order = leaves_list(optimal_leaf_ordering(col_linkage, matrix.T))
    return {
        'row_linkage': row_linkage,
        'col_linkage': col_linkage,
        'row_order': row_order,
        'col_order': col_order
    }

def correlation_matrix(matrix: np.ndarray, method: str = 'spearman', axis: str = 'feature'):
    """
    Generates a feature-feature or sample-sample correlation matrix from the data, defaults to spearman because 
    metabolomics data is not typically normally distributed and spearman is rank-based

    Params
    ------
        matrix                      matrix of values to test
        method                      'spearman' or 'pearson', how we calculate similarity
        axis                        'featur' or 'sample' depending on which we want to find correaltion between

    Returns
    -------
        {
        corr: (n_features, n_features) or (n_samples, n_samples) - correlation matrix, values in range [-1,1]
        pvals: (n_features, n_features) or (n_samples, n_samples) - p-values for each corerlation, only
               recorded for spearman, pearson is None, entry[i,j] is the probability that correlation is by chance,
               apply FDR correlation (if many features) before thresholding to <0.05 to find significanlty correlated
               entries
        }
    """
    # ensure methods are correctly specified
    if method not in ['spearman', 'pearson']:
        logger.warning(f"Unknown correlation matrix method {method} supplied, defaulting to spearman")
        method = 'spearman'
    if axis not in ['feature', 'sample']:
        logger.warning(f"unknown axis for correlation matrix {axis}, defaulting to feature")
        axis = 'feature'

    # format matrix
    if axis == 'sample':
        mat = matrix
    else:
        mat = matrix.T
    n = mat.shape[0]
    
    # conduct testing
    if method == 'spearman':
        corr, pvals = spearmanr(mat)
    elif method == 'pearson':
        corr = np.zeros((n,n))
        pvals = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                corr[i,j],pvals[i,j] = pearsonr(mat[i],mat[j])
    
    return {'corr': corr, 'pvals': pvals}

# endregion

# region                 ---------- Feature Selection ----------

def vip_scores(model):
    """
    Variable Importance in Projection from a fitted PLSRegression model

    Params
    ------
    model                           PLS-DA model for VIP analysis

    Returns
    -------
        vips                        (n_features) array where higher values indicate features that contribute
                                    more to group separation. VIP > 1.0 is the standard threshold for a feature
                                    being considered important.
    """
    t = model.x_scores_           # sample scores
    w = model.x_weights_          # feature weights
    q = model.y_loadings_         # y loadings
    p, h = w.shape                # n_features, n_components

    vips = np.zeros(p)
    s = np.diag(t.T @ t @ q.T @ q)
    total = np.sum(s)

    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * np.sum(s * weight) / total)

    return vips

def significant_features(p_values: np.ndarray, fold_changes: np.ndarray, p_thresh: float = 0.05, fc_thresh: float = 1.0):
    """
    Filters features by p-value and absolute log2 fold change threshold to find significantly different features
    between groups

    Params
    ------
        p_values                    array of p values from some sort of analysis
        fold_changes                fold changes between groups index matched to p values
        p_thresh                    threshold for significance in P values
        fc_thresh                   threshold for significance in fold change

    Returns
    -------
        indices: (n_significant,) — column indices of significant features
        p_values: (n_significant,) — p-values for significant features
        fold_changes: (n_significant,) — fold changes for significant features
    """
    mask = (p_values < p_thresh) & (np.abs(fold_changes) >= fc_thresh)
    indices = np.where(mask)[0]
    return {
        'indices': indices,
        'p_values': p_values[indices],
        'fold_changes': fold_changes[indices]
    }

def top_correlated(corr_matrix: np.ndarray, entry_idx: int, n: int = 10):
    """
    Finds the n most positively and negatively correlated entries to the given entry

    Params
    ------
        corr_matrix                     correlation matrix for analysis
        entry_idx                       index of entry we want to find top correlated things for
        n                               number of top correlated matches to return

    Returns
    -------
        {
        positive: (n) indices of top n positively correlated entries
        negative: (n) indices of top n negatively correlated entries
        pos_values: (n) correlation values for positive hits
        neg_values: (n) correlation values for negative hits
        }
    """
    row = corr_matrix[entry_idx].copy()
    row[entry_idx] = 0  # exclude self-correlation

    pos_idx = np.argsort(row)[::-1][:n]
    neg_idx = np.argsort(row)[:n]

    return {
        'positive': pos_idx,
        'negative': neg_idx,
        'pos_values': row[pos_idx],
        'neg_values': row[neg_idx]
    }

def correlated_to_feature(corr_matrix: np.ndarray, feature_idx: int, threshold: float = 0.7):
    """
    Returns all features correlated above a given threshold to a specified feature.

    Params
    ------
        corr_matrix                     correlation matrix for analysis
        feature_idx                     index of feature to find correlated features for
        threshold                       correlation threshold for returning hits

    Returns
    -------
        {
        indices: array of indices of correlated features
        values: array correlation values for those features
        }
    """
    row = corr_matrix[feature_idx].copy()
    row[feature_idx] = 0  # exclude self

    mask = np.abs(row) >= threshold
    indices = np.where(mask)[0]

    return {
        'indices': indices,
        'values': row[indices]
    }

# endregion
