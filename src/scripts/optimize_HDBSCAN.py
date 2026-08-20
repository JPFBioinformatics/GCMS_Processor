
# region Imports

import numpy as np
from pathlib import Path
import json
import itertools
from matplotlib.backends.backend_pdf import PdfPages
import hdbscan
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

from src.intensity_matrix import IntensityMatrix as IM
from src.scripts.helpers import (quantile_normalization, normalize_matrix, rolling_median_2d, 
                                 plot_gs_heatmaps, plot_histogram)

# endregion

# region logging

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=Path(__file__).parent / "optimize_HDBSCAN.log"
)
logger = logging.getLogger(__name__)

# endregion

iter = 1
n_samples = 250_000

per_row = False
ion = 147

include_cwt_data = False

knn_info = False

json_path = Path(f'im_data.json')
out_path = Path(f'HDBSCAN_optimization_{int(n_samples/1000)}k_{iter}.pdf')

# region data loading

if not json_path.exists():
    raise ValueError('No json data found')

with open(json_path, 'r') as f:
    data = json.load(f)

    # endregion

# Build a three row matrix to represent each point as a column of 1d, 2d, and signal
if per_row:

    # check to see that the mz is vaild
    mzs = data['mzs']
    if ion not in mzs:
        raise ValueError('Ion not detected')
    ion_idx = mzs.index(ion)

    # normalize individual rows
    cwt_max_scores = np.array(data['cwt_max_scores'][ion_idx], dtype=float)
    cwt_max_scores_norm = quantile_normalization(cwt_max_scores)

    cwt_max_scales = np.array(data['cwt_max_scales'][ion_idx], dtype=float)
    cwt_max_scales_norm = normalize_matrix(cwt_max_scales)

    first_derivs = np.array(data['first_derivs'][ion_idx], dtype=float)
    fd_trend = rolling_median_2d(first_derivs, window=51)
    first_derivs_norm = normalize_matrix(first_derivs-fd_trend)

    second_derivs = np.array(data['second_derivs'][ion_idx], dtype=float)
    second_derivs_norm = normalize_matrix(second_derivs)

    smoothed_signal = np.array(data['smoothed_signal'][ion_idx], dtype=float)
    sm_trend = rolling_median_2d(smoothed_signal, window=51)
    smoothed_signal_norm = normalize_matrix(smoothed_signal-sm_trend)

    # stack to data matrix
    if include_cwt_data:
        X = np.column_stack([
            first_derivs_norm,
            second_derivs_norm,
            smoothed_signal_norm,
            cwt_max_scales_norm,
            cwt_max_scores_norm,
        ])
    else:
        X = np.column_stack([
            first_derivs_norm,
            second_derivs_norm,
            smoothed_signal_norm,
        ])

else:

    # normalize individual matrices
    cwt_max_scores = np.array(data['cwt_max_scores'], dtype=float)
    cwt_max_scores_norm = quantile_normalization(cwt_max_scores)

    cwt_max_scales = np.array(data['cwt_max_scales'], dtype=float)
    cwt_max_scales_norm = normalize_matrix(cwt_max_scales)

    first_derivs = np.array(data['first_derivs'], dtype=float)
    fd_trend = rolling_median_2d(first_derivs, window=51)
    first_derivs_norm = normalize_matrix(first_derivs-fd_trend)

    second_derivs = np.array(data['second_derivs'], dtype=float)
    second_derivs_norm = normalize_matrix(second_derivs)

    smoothed_signal = np.array(data['smoothed_signal'], dtype=float)
    sm_trend = rolling_median_2d(smoothed_signal, window=51)
    smoothed_signal_norm = normalize_matrix(smoothed_signal-sm_trend)

    # stack to data matrix
    if include_cwt_data:
        X = np.column_stack([
            first_derivs_norm.ravel(),
            second_derivs_norm.ravel(),
            smoothed_signal_norm.ravel(),
            cwt_max_scales_norm.ravel(),
            cwt_max_scores_norm.ravel()
        ])
    else:
        X = np.column_stack([
            first_derivs_norm.ravel(),
            second_derivs_norm.ravel(),
            smoothed_signal_norm.ravel(),
        ])

# region data processing

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
print('\n')

# sample for dbscan, plotting, and pca
n_samples = 250_000
eps = 50
sample_idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)

# sample X for PCA/DBSCAN
x_sample = X[sample_idx]

# print out the neighbor count information
raw_score_sample = cwt_max_scores.ravel()[sample_idx]
nbrs = NearestNeighbors(radius=22).fit(x_sample[:5000])
results = nbrs.radius_neighbors(x_sample[:100], return_distance=False)
neighbor_counts = [len(idx) for idx in results]
max_neighbors = np.nanmax(neighbor_counts)
med_neighbors = np.nanmedian(neighbor_counts)
mad_neighbors = np.nanmedian(abs(neighbor_counts-med_neighbors))
min_neighbors = np.nanmin(neighbor_counts)

neighbor_title = f'Neighbor Counts max={max_neighbors} min={min_neighbors} med={med_neighbors} MAD={mad_neighbors}'

# gridsearched DBSCAN
logger.info("HDBSCAN gridsesarch innitiated")
param_grid = {
    'min_cluster_size': [500, 1000,2000,3000,4000],
    'min_samples': [20, 50, 100, 150, 200],
    'eps': [0, 1, 2]
}
gs_results = []
for mcs, ms, cse in itertools.product(param_grid['min_cluster_size'],param_grid['min_samples'],param_grid['eps']):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_epsilon=cse, 
                                gen_min_span_tree=True, prediction_data=False)
    test_labels = clusterer.fit_predict(x_sample)

    n_clusters = len(set(test_labels)) - (1 if -1 in test_labels else 0)
    noise_frac = (test_labels == -1).sum() / len(test_labels)
    validity = getattr(clusterer, 'relative_validity_', None) # DBCV-like internal metric

    gs_results.append((mcs, ms, cse, n_clusters, noise_frac, validity))

logger.info('HDBSCAN gridsearch Completed')

# endregion

with PdfPages(out_path) as pdf:

    if knn_info:
        plot_histogram(pdf, neighbor_title, xlabel='Neighbor Count', values=neighbor_counts)

    # HDBSCAN GS results
    plot_gs_heatmaps(pdf, param_grid, gs_results)
    logger.info('GS Heatmaps Plotted')
