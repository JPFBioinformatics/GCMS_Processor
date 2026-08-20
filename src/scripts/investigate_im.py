
# region Imports

import numpy as np
from pathlib import Path
import json
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
import hdbscan
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

from src.intensity_matrix import IntensityMatrix as IM
from src.scripts.helpers import (quantile_normalization, normalize_matrix, rolling_median_2d, 
                                 plot_histogram, plot_k_distance, plot_cluster_hexbin_facets,
                                 plot_multicluster_hexbins, plot_joint_scatter, plot_skree, 
                                 plot_table, adaptive_linthresh, plot_hexbin)

# endregion

# region logging

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=Path(__file__).parent / "investigate_im.log"
)
logger = logging.getLogger(__name__)

# endregion

# iteration counter and number of samples
iter = 1
n_samples = 250_000

# HDBSCAN parameters
mcs = 500
ms = 100
cse = 1

# paths
json_path = Path(f'im_data.json')
out_path = Path(f'HDBSCAN_optimization_{n_samples/1000}k_{iter}.pdf')

# region data loading

if not json_path.exists():
    raise ValueError("No json data found")

with open(json_path, 'r') as f:
    data = json.load(f)

logger.info(f"Data Loaded")

# endregion

# region Processing

# get data for top10 highest height threshold rows
top10_hts = np.argsort(data['height_thresholds'])[-10:][::-1]
top10_mzs = [data['mzs'][i] for i in top10_hts]
top10_mads = [data['height_thresholds'][i] for i in top10_hts]
height_data = []
for ion,mad in zip(top10_mzs, top10_mads):
    height_data.append([ion,mad])

# normalize matrices

cwt_max_scores = np.array(data['cwt_max_scores'], dtype=float)
cwt_max_scores_norm = quantile_normalization(cwt_max_scores)

cwt_max_scales = np.array(data['cwt_max_scales'], dtype=float)
cwt_scales_norm = normalize_matrix(cwt_max_scales)

first_derivs = np.array(data['first_derivs'], dtype=float)
fd_trend = rolling_median_2d(first_derivs, window=51)
first_derivs_norm = normalize_matrix(first_derivs-fd_trend)

second_derivs = np.array(data['second_derivs'], dtype=float)
second_derivs_norm = normalize_matrix(second_derivs)

smoothed_signal = np.array(data['smoothed_signal'], dtype=float)
sm_trend = rolling_median_2d(smoothed_signal, window=51)
smoothed_signal_norm = normalize_matrix(smoothed_signal-sm_trend)

logger.info("Matrices normalized")

# get shape of the matrices
n_rows,n_cols = first_derivs_norm.shape

# Build a three row matrix to represent each point as a column of 1d, 2d, and signal
X = np.column_stack([
    first_derivs_norm.ravel(),
    second_derivs_norm.ravel(),
    smoothed_signal_norm.ravel(),
    #cwt_scales_norm.ravel(),
    #cwt_scores_norm.ravel()
])
names = ['First Derivatives', 'Second Derivatives', 'Smoothed Signal', 'CWT Scales', 'CWT Scores']
matrices = [first_derivs, second_derivs, smoothed_signal, cwt_max_scales, cwt_max_scores]
mads = []
for i,matrix in enumerate(matrices):
    name = names[i]
    med = np.nanmedian(matrix, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(matrix - med), axis=1, keepdims=True) * 1.4826
    zoomed = mad[(mad <= 1)]
    mads.append(zoomed)
    print(f"\n\nFeature: {name}")
    print(f"Min MAD across rows: {mad.min()}")
    print(f"Absolute Min Median across rows: {abs(med).min()}")
    print(f"Rows with mad < 1e-6: {(mad<1e-6).sum()}")

# sample for dbscan, plotting, and pca
n_samples = 250_000
eps = 50
sample_idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)

# sample X for PCA/DBSCAN
x_sample = X[sample_idx]
raw_score_sample = cwt_max_scores.ravel()[sample_idx]

# sample scores/scales for plotting
scores_flat = cwt_max_scores.ravel()
s_scores = scores_flat[sample_idx]

scales_flat = cwt_max_scales.ravel()
s_scales = scales_flat[sample_idx]

# preform primary HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_epsilon=cse, prediction_data=True)
sample_labels = clusterer.fit_predict(x_sample)
labels, strengths = hdbscan.approximate_predict(clusterer, X)
logger.info("HDBSCAN complete")

# preform PCA
n_components = X.shape[1]
pca = PCA(n_components=n_components)
scores = pca.fit_transform(x_sample)
explained_variance_ratio = pca.explained_variance_ratio_
loadings = pca.components_

# apply pca to entire X matrix
full_scores = pca.transform(X)
logger.info("PCA Complete")

# endregion

with PdfPages(out_path) as pdf:

    for i in range(len(mads)):
        if len(mads[i]) == 0:
            continue
        nonzero = mads[i][mads[i] > 0]
        linthresh = nonzero.min() if len(nonzero) > 0 else 1e-9

        plot_histogram(pdf, title=f'{names[i]} per-ion signal MAD distribution (n={len(mads[i])})',
                        xlabel='MAD', values=mads[i], symlog=True, linthresh=linthresh, rotate_labels=True)

    # d_distance plot 
    plot_k_distance(pdf, min_samples=20, x_sample=x_sample, trim_pct=99.5, include_table=True, feature_names=names, raw_score_sample=raw_score_sample)
    logger.info('K Distances Plotted')

    feature_names = ['First Deriv (norm)', 'Second Deriv (norm)', 'Signal (norm)'] #, 'CWT Scale (norm)', 'CWT Score (norm)']

    # plot pariwise scatter plots of first/second deriv and signal (all normalized)
    features = [first_derivs_norm.ravel(), second_derivs_norm.ravel(), smoothed_signal_norm.ravel()] #, cwt_scales_norm.ravel(), cwt_scores_norm.ravel()]
    for i,j in combinations(range(3),2):
        plot_cluster_hexbin_facets(
            pdf, f"HDBSCAN Clusters: {feature_names[j]} vs {feature_names[i]}",
            feature_names[i], feature_names[j],
            features[i], features[j], labels.ravel()
        )

    # plot multicluster hexbins/scatter on PCA axes
    plot_multicluster_hexbins(pdf, "HDBSCAN Clusters (PCA, full dataset)", "PC1", "PC2",
                                full_scores[:,0], full_scores[:,1], labels)
    plot_joint_scatter(pdf, "HDBSCAN Clusters (PCA, full dataset)", "PC1", "PC2",
                        full_scores[:,0], full_scores[:,1], labels, max_points_per_cluster=20_000)
    plot_skree(pdf, explained_variance_ratio)
    
    table_data = [
        [name, f"{loadings[0,i]:.3f}", f"{loadings[1,i]:.3f}"]
        for i,name in enumerate(feature_names)
    ]
    plot_table(pdf, "PCA Feature Loadings (PC1 & PC2)", table_data, ['Feature', 'PC1 Loading', 'PC2 Loading'])

    logger.info("Clustering Plotted")
    
    # plot tables
    plot_table(pdf, "Maximum Median Absolute Deviation by Ion", height_data, ['Ion', 'MAD'])
    logger.info("Table Plotted")

    # plot histograms
    plot_histogram(pdf, "S/N Ratios", "S/N Ratio", data['sn_ratios'], symlog=True)
    logger.info('SN ratio histogram plotted')
    plot_histogram(pdf, "Peak CWT Scores", "Score", data['cwt_scores'], symlog=True)
    logger.info('Peak CWT Scores histogram plotted')
    plot_histogram(pdf, "Peak CWT Scales", "Scales", data['cwt_scales'], bin_size=1)
    logger.info('Peak CWT Scales histogram plotted')
    plot_histogram(pdf, 'Raw CWT max Score Distribution', 'Score', s_scores, symlog=True)
    logger.info('Raw CWT max Score Distribution histogram plotted')
    plot_histogram(pdf, 'Raw CWT Max Scale Distribution', 'Scale', s_scales, bin_size=1)
    logger.info('Raw CWT Max Scale Distribution histogram plotted')
    plot_histogram(pdf, "Ridge Spans", "Ridge Span", data['ridge_spans'], bin_size=1)
    logger.info('Ridge Spans histogram plotted')
    plot_histogram(pdf, "Heights", "Height", data['heights'], symlog=True)
    logger.info('Heights histogram plotted')
    plot_histogram(pdf, "Height Thresholds", "Height Threshold", data['height_thresholds'], symlog=True, linthresh=adaptive_linthresh(data['height_thresholds']))
    logger.info('Height Thresholds histogram plotted')
    plot_histogram(pdf, "Widths", "Width", data['widths'], bin_size=1)
    logger.info('Widths histogram plotted')

    logger.info("All Histograms Plotted")

    # plot a v b scatter plots
    x_scale = 'linear'
    y_scale = 'linear'
    excluded_keys = ['height_thresholds', 'mzs', 'widths', 'first_derivs', 'second_derivs', 
                     'smoothed_signal', 'cwt_max_scores', 'cwt_max_scales']
    keys = [key for key in data.keys() if key not in excluded_keys]
    for key1, key2 in combinations(keys,2):

            symlog_metrics = ['cwt_scores', 'sn_ratios', 'heights']

            x_scale = 'symlog' if key1 in symlog_metrics else 'linear'
            y_scale = 'symlog' if key2 in symlog_metrics else 'linear'

            x_data = data[key1]
            y_data = data[key2]

            title = f"{key2} vs {key1}"

            plot_hexbin(pdf, title, key1, key2, x_data, y_data, xscale=x_scale, yscale=y_scale, gridsize=50)

            x_scale = 'linear'
            y_scale = 'linear'

    logger.info("A vs B scatter plots completed")