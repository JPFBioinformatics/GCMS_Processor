
# region Imports

import numpy as np
from pathlib import Path
import json
import itertools
from matplotlib.backends.backend_pdf import PdfPages
import hdbscan
from sklearn.neighbors import NearestNeighbors

from src.scripts.helpers import (quantile_normalization, normalize_matrix, rolling_median_2d, 
                                 plot_gs_heatmaps, plot_histogram, adaptive_linthresh)

# endregion

# region logging

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=Path(__file__).parent / "logs" / "optimize_HDBSCAN.log"
)
logger = logging.getLogger(__name__)

# endregion

n_samples = 250_000
seed = 42

param_grid = {
    'min_cluster_size': [100, 200, 300, 400, 500],
    'min_samples': [20, 50, 100, 150, 200],
    'eps': [1]
}

per_row = False
ion = 147

include_cwt_data = False

knn_info = False

threshold = False

root_dir = Path(__file__).resolve().parents[2]
out_path = root_dir / 'peak_metrics_data'

if threshold:

    # get input path
    json_path = Path(f'im_data_threshold.json')

    # build output path
    if include_cwt_data:
        out_path = out_path / '5_embedding'
    else:
        out_path = out_path / '3_embedding'

    out_path = out_path / 'threshold_true'

    if per_row:
        out_path = out_path / 'per_row'
    else:
        out_path = out_path / 'full_matrix'
    
else:

    # get input path
    json_path = Path(f'im_data_no_threshold.json')

    # build output path
    if include_cwt_data:
        out_path = out_path / '5_embedding'
    else:
        out_path = out_path / '3_embedding'

    out_path = out_path / 'threshold_false'

    if per_row:
        out_path = out_path / 'per_row'
    else:
        out_path = out_path / 'full_matrix'

file_name = f'HDBSCAN_optimization_{int(n_samples/1000)}k_{seed}seed.pdf'
out_path = out_path / file_name

# region data loading

if not json_path.exists():
    raise ValueError('No json data found')

with open(json_path, 'r') as f:
    data = json.load(f)

    # endregion

# Build a embedding matrix to represent each point as a column of 1d, 2d, and signal
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
        names = ['First Derivatives', 'Second Derivatives', 'Smoothed Signal', 'CWT Scales', 'CWT Scores']
    else:
        X = np.column_stack([
            first_derivs_norm,
            second_derivs_norm,
            smoothed_signal_norm,
        ])
        names = ['First Derivatives', 'Second Derivatives', 'Smoothed Signal']

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
        names = ['First Derivatives', 'Second Derivatives', 'Smoothed Signal', 'CWT Scales', 'CWT Scores']
    else:
        X = np.column_stack([
            first_derivs_norm.ravel(),
            second_derivs_norm.ravel(),
            smoothed_signal_norm.ravel(),
        ])
        names = ['First Derivatives', 'Second Derivatives', 'Smoothed Signal']

# region Data Processing

# find how much duplication there is in X
# vals=(n_unique, n_features) list of unique features
# inverse=(n_rows) list of which val this row belongs to
# counts=(n_unique,) counts of each unique val
vals, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)

# find the domianant index value
dominant_idx = counts.argmax()
# if there is a dominant group mask it for sampling
if counts[dominant_idx]/len(X) >= 0.1:
    noise_mask = (inverse == dominant_idx)
    noise_indices = np.where(noise_mask)[0]
    signal_indices = np.where(~noise_mask)[0]
# if no dominant group then don't mask it
else:
    noise_indices = np.array([], dtype=int)
    signal_indices = np.arange(len(X))

# sampling
rng = np.random.default_rng(seed)
sample_idx = rng.choice(signal_indices, size=min(n_samples, len(signal_indices)), replace=False)

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

    # plot distributions of X 
    for i, metric in enumerate(names):

        # get metrics
        current_row = X[:,i]
        median = np.nanmedian(current_row)
        mad = np.nanmedian(np.abs(current_row-median))
        row_max = np.nanmax(current_row)
        row_min = np.nanmin(current_row)
        title = f'{metric} distribution\nMed={median:.2f}; MAD={mad:.2f}; Max={row_max:.2f}; Min={row_min:.2f}'

        # get formatting for histogram
        if metric in ['First Derivatives', 'Second Derivatives', 'Smoothed Signal', 'CWT Scales']:
            symlog = True
            bin_size = None
            linthresh = adaptive_linthresh(current_row)
            rotate_labels = True
        else:
            symlog = False
            bin_size = 1
            linthresh = 1.0
            rotate_labels = False

        # plot histogram
        plot_histogram(pdf, title, xlabel=f'{metric}', values=current_row, symlog=symlog,
                       bin_size=bin_size, linthresh=linthresh, rotate_labels=rotate_labels)
        logger.info(f'{metric} histogram plotted')