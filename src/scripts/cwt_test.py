
# region Imports

import numpy as np
from pathlib import Path
from datetime import datetime
from src.mzml_processor import create_scan_matrix
from src.config_loader import ConfigLoader
from src.plotting import plot_heatmap
import matplotlib.pyplot as plt
from src.utils import sanitize_name
import json
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from numpy.lib.stride_tricks import sliding_window_view

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)


# endregion

# region logging

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=Path(__file__).parent / "cwt_test.log"
)
logger = logging.getLogger(__name__)

# endregion

def symlog_bins(values, linthresh=1.0, n_bins=30):
    values = np.asarray(values)
    pos = values[values > linthresh]
    neg = values[values < -linthresh]

    edges = [np.linspace(-linthresh, linthresh, 5)]
    if len(neg) > 0:
        edges.insert(0, -np.geomspace(linthresh, abs(neg.min()), n_bins)[::-1])
    if len(pos) > 0:
        edges.append(np.geomspace(linthresh, pos.max(), n_bins))

    return np.unique(np.concatenate(edges))

def plot_histogram(pdf: PdfPages, title: str, xlabel: str, values, 
                   bin_size: int = None, n_bins: int = None, 
                   symlog: bool = False, linthresh: float = 1.0):
    
    fig, ax = plt.subplots()

    if symlog:
        bins = symlog_bins(values, linthresh=linthresh, n_bins = n_bins or 30)
        ax.set_xscale('symlog', linthresh=linthresh)
    elif n_bins is not None:
        bins = n_bins
    elif bin_size is not None:
        bins = np.arange(min(values), max(values) + bin_size, bin_size)
    else:
        bins = 'auto'

    ax.hist(values, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)

    pdf.savefig(fig)
    plt.close(fig)

def plot_scatter(pdf, title, xlabel, ylabel, x_values, y_values,
                  color_values=None, color_label=None,
                  xscale='linear', yscale='linear',
                  x_linthresh=1.0, y_linthresh=1.0, alpha=0.5):
    
    fig, ax = plt.subplots()

    if color_values is not None:
        sc = ax.scatter(x_values, y_values, c=color_values, alpha=alpha, rasterized=True)
        cbar = fig.colorbar(sc, ax=ax)
        if color_label:
            cbar.set_label(color_label)
    else:
        ax.scatter(x_values, y_values, alpha=alpha, rasterized=True)

    if xscale == 'symlog':
        ax.set_xscale('symlog', linthresh=x_linthresh)
    else:
        ax.set_xscale(xscale)

    if yscale == 'symlog':
        ax.set_yscale('symlog', linthresh=y_linthresh)
    else:
        ax.set_yscale(yscale)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    pdf.savefig(fig)
    plt.close(fig)

def plot_cluster_scatter(pdf, title, xlabel, ylabel, x_values, y_values,
                         labels, alpha=0.5):
    
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    labels = np.asarray(labels)

    fig,ax = plt.subplots()
    cmap = plt.get_cmap('tab10')

    for i, lbl in enumerate(np.unique(labels)):
        mask = labels == lbl
        if lbl == -1:
            ax.scatter(x_values[mask], y_values[mask], color='lightgrey', s=5,
                       alpha=alpha/2, label=f"Noise (n={mask.sum()})", rasterized=True)
        else:
            ax.scatter(x_values[mask], y_values[mask], color=cmap(i%10), s=5,
                       alpha=alpha, label=f"Cluster {lbl} (n={mask.sum()})", rasterized=True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=8, loc='best')

    pdf.savefig(fig)
    plt.close(fig)

def plot_hexbin(pdf, title, xlabel, ylabel, x_values, y_values,
                xscale = 'linear', yscale = 'linear', x_linthresh = 1.0,
                y_linthresh = 1.0, gridsize = 50, cmap = 'viridis'):
    
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    if xscale == 'symlog':
        x_plot = np.sign(x_values) * np.log1p(np.abs(x_values) / x_linthresh)
        xlabel = f"{xlabel} (symlog)"
    else:
        x_plot = x_values

    if yscale == 'symlog':
        y_plot = np.sign(y_values) * np.log1p(np.abs(y_values) / y_linthresh)
        ylabel = f"{ylabel} (symlog)"
    else:
        y_plot = y_values

    fig, ax = plt.subplots()
    hb = ax.hexbin(x_plot, y_plot, gridsize=gridsize, cmap=cmap, mincnt=1)
    cbar =fig.colorbar(hb, ax=ax)
    cbar.set_label('Count')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    pdf.savefig(fig)
    plt.close(fig)

def plot_cluster_hexbin_facets(pdf, title, xlabel, ylabel, x, y, labels, gridsize=50, cmap='viridis'):

    unique_labels = np.unique(labels)
    ncols = min(3, len(unique_labels))
    nrows = int(np.ceil(len(unique_labels) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)

    for ax, lbl in zip(axes.flat, unique_labels):
        mask = labels == lbl
        ax.hexbin(x[mask], y[mask], gridsize=gridsize, cmap=cmap, mincnt=1)
        name = 'Noise' if lbl == -1 else f'Cluster {lbl}'
        ax.set_title(f"{name} (n={mask.sum()})", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)

    for ax in axes.flat[len(unique_labels):]:
        ax.axis('off')

    fig.suptitle(title)
    fig.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)

def plot_table(pdf, title, data, collabels):

    fig,ax = plt.subplots()
    ax.axis('off')

    cell_text = [[str(cell) for cell in row] for row in data]
    table = ax.table(
        cellText=cell_text,
        colLabels=collabels,
        loc='center',
        cellLoc='left'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(2)))

    ax.set_title(title)

    pdf.savefig(fig)
    plt.close(fig)

def rolling_median_2d(arr, window=51):
    pad = window//2
    padded = np.pad(arr, ((0,0), (pad,pad)), mode='edge')
    windows = sliding_window_view(padded, window_shape=window, axis=1)
    return np.nanmedian(windows,axis=-1)

def normalize_matrix(arr, axis=1, eps=1e-9):
    med = np.nanmedian(arr, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(arr-med), axis=axis, keepdims=True) * 1.4826
    return (arr-med) / (mad+eps)

name = "deriv_testing"

json_path = Path(f'peak_metrics_{name}.json')
out_path = Path(f'peak_metrics_{name}.pdf')

if json_path.exists():

    with open(json_path, 'r') as f:
        data = json.load(f)

else:

    file_path = Path(r"C:\Jack\Projects\IlyaAura Mouse Labelling\7_10_int1\mzML_files\Int 1.mzML")

    cfg_path = Path(__file__).parent / 'config.yaml'
    cfg = ConfigLoader(cfg_path)
    peak_mode = cfg.get('peak_mode')

    starttime = datetime.now()

    im = create_scan_matrix(file_path, cfg=cfg)

    endtime = datetime.now()

    # get data for histograms
    sn_ratios = []
    cwt_scores = []
    cwt_scales = []
    ridge_spans = []
    heights = []
    scales = []
    scores = []
    widths = []
    height_thresholds = im.height_thresholds
    for _,peak_list in im.peak_dict.items():
        for peak in peak_list:
            sn_ratios.append(float(peak['sn_ratio']))
            cwt_scores.append(float(peak['cwt_score']))
            cwt_scales.append(float(peak['cwt_scale']))
            ridge_spans.append(float(peak['ridge_span']))
            heights.append(float(peak['height']))
            widths.append(float(peak['right_bound'] - peak['left_bound']))

    data = {
        'sn_ratios': sn_ratios,
        'cwt_scores': cwt_scores,
        'cwt_scales': cwt_scales,
        'ridge_spans': ridge_spans,
        'heights': heights,
        'widths': widths,
        'height_thresholds': height_thresholds,
        'mzs': [int(mz) for mz in im.unique_mzs],
        'first_derivs': im.first_derivs.tolist(),
        'second_derivs': im.second_derivs.tolist(),
        'smoothed_signal': im.smoothed_signal.tolist()
    }
    with open(json_path, 'w') as f:
        json.dump(data, f)

# get data for top10 highest height threshold rows
top10_hts = np.argsort(data['height_thresholds'])[-10:][::-1]
top10_mzs = [data['mzs'][i] for i in top10_hts]
top10_mads = [data['height_thresholds'][i] for i in top10_hts]
height_data = []
for ion,mad in zip(top10_mzs, top10_mads):
    height_data.append([ion,mad])

# normalize first deriv, second deriv, and smoothed signal matrices for DBSCAN
first_derivs = np.array(data['first_derivs'], dtype=float)
second_derivs = np.array(data['second_derivs'], dtype=float)
smoothed_signal = np.array(data['smoothed_signal'], dtype=float)

fd_trend = rolling_median_2d(first_derivs, window=51)
first_derivs_norm = normalize_matrix(first_derivs-fd_trend)

second_derivs_norm = normalize_matrix(second_derivs)

sm_trend = rolling_median_2d(smoothed_signal, window=51)
smoothed_signal_norm = normalize_matrix(smoothed_signal-sm_trend)

# get shape of the matrices
n_rows,n_cols = first_derivs_norm.shape

# Build a three row matrix to represent each point as a column of 1d, 2d, and signal
X = np.column_stack([
    first_derivs_norm.ravel(),
    second_derivs_norm.ravel(),
    smoothed_signal_norm.ravel()
])

# fit DBSCAN on a subset
n_samples = 100_000
sample_idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
x_sample = X[sample_idx]

db = DBSCAN(eps=0.5, min_samples=20)
sample_labels = db.fit_predict(x_sample)

# extend subset to full dataset with KNN
trained_mask = sample_labels != -1
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_sample[trained_mask], sample_labels[trained_mask])
labels = knn.predict(X)

# identify largest non-noise cluster as baseline
vals,counts = np.unique(labels[labels != 1], return_counts=True)
baseline_label = vals[np.argmax(counts)]
baseline_mask = (labels == baseline_label).reshape(n_rows,n_cols)

with PdfPages(out_path) as pdf:

    # plot pariwise scatter plots of first/second deriv and signal (all normalized)
    feature_names = ['First Deriv (norm)', 'Second Deriv (norm)', 'Signal (norm)']
    features = [first_derivs_norm.ravel(), second_derivs_norm.ravel(), smoothed_signal_norm.ravel()]
    for i,j in combinations(range(3),2):
        plot_cluster_hexbin_facets(
            pdf, f"DBSCAN Clusters: {feature_names[j]} vs {feature_names[i]}",
            feature_names[i], feature_names[j],
            features[i], features[j], labels.ravel()
        )
    
    # plot tables
    plot_table(pdf, "Maximum Median Absolute Deviation by Ion", height_data, ['Ion', 'MAD'])

    # plot histograms
    plot_histogram(pdf, "S/N Ratios", "S/N Ratio", data['sn_ratios'], symlog=True)
    plot_histogram(pdf, "CWT Scores", "Score", data['cwt_scores'], symlog=True)
    plot_histogram(pdf, "CWT Scales", "Scales", data['cwt_scales'], bin_size=1)
    plot_histogram(pdf, "Ridge Spans", "Ridge Span", data['ridge_spans'], bin_size=1)
    plot_histogram(pdf, "Heights", "Height", data['heights'], symlog=True)
    plot_histogram(pdf, "Height Thresholds", "Height Threshold", data['height_thresholds'], symlog=True)
    plot_histogram(pdf, "Widths", "Width", data['widths'])

    # plot a v b scatter plots
    x_scale = 'linear'
    y_scale = 'linear'
    excluded_keys = ['height_thresholds', 'mzs', 'widths', 'first_derivs', 'second_derivs', 'smoothed_signal']
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
