
# region Imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from collections import Counter, defaultdict

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
                   symlog: bool = False, linthresh: float = 1.0, 
                   rotate_labels: bool = False):
    
    fig, ax = plt.subplots()

    if symlog:
        bins = symlog_bins(values, linthresh=linthresh, n_bins = n_bins or 30)
        ax.set_xscale('symlog', linthresh=linthresh)
        if np.all(np.asarray(values) >= 0):
            ax.set_xlim(left=0)
    elif n_bins is not None:
        bins = n_bins
    elif bin_size is not None:
        vmin,vmax = min(values), max(values)
        if vmin == vmax:
            bins = [vmin - bin_size /2, vmin + bin_size / 2]
        else:
            bins = np.arange(min(values), max(values) + bin_size, bin_size)
    else:
        bins = 'auto'
    if rotate_labels:
        plt.setp(ax.get_xticklabels(), rotation=50, ha='right')

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

def plot_multicluster_hexbins(pdf, title, xlabel, ylabel, x, y, labels, cmaps=None,
                              gridsize=50, mincnt=1, alpha=0.8):

    # convert to arrays
    x = np.asarray(x); y = np.asarray(y); labels = np.asarray(labels)

    # define colormaps
    default_cmaps = ['Purples', 'Reds', 'Blues', 'Greens', 'Oranges']

    # plot figure
    fig,ax = plt.subplots(figsize=(8,6))

    # order clusters largest to smallest
    unique_labels, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    unique_labels = unique_labels[order]

    # plot individual clusters
    cluster_idx = 0
    for lbl in unique_labels:
        mask = labels == lbl
        if lbl == -1:
            cmap,name = 'Greys', f'Noise (n={mask.sum()})'
        else:
            cmap = (cmaps or default_cmaps)[cluster_idx % len(cmaps or default_cmaps)]
            name = f"Cluster {lbl} (n={mask.sum()})"
            cluster_idx += 1

        hb = ax.hexbin(x[mask],y[mask], gridsize=gridsize, cmap=cmap, mincnt=mincnt,
                       alpha=alpha)
        cbar = fig.colorbar(hb, ax=ax, shrink=0.25, pad=0.05)
        cbar.set_label(name, fontsize=6)
        cbar.ax.tick_params(labelsize=5)

    # set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    pdf.savefig(fig)
    plt.close(fig)

def plot_joint_scatter(pdf, title, xlabel, ylabel, x, y, labels, cmaps=None, alpha=0.7, 
                       s=8, max_points_per_cluster=20_00):

    # convert to arrays
    x = np.asarray(x); y = np.asarray(y); labels = np.asarray(labels)

    # define colormaps
    default_cmaps = ['Purples', 'Reds', 'Blues', 'Greens', 'Oranges']

    # setup figure
    fig = plt.figure(figsize=(12,12))
    gs = fig.add_gridspec(2,2, width_ratios=(4,1), height_ratios=(1,4), wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1,0])
    ax_top = fig.add_subplot(gs[0,0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1,1], sharey=ax_main)

    # reorder labels from most to least
    unique_labels, counts = np.unique(labels, return_counts=True)
    unique_labels = unique_labels[np.argsort(-counts),]

    x_grid = np.linspace(x.min(), x.max(), 200)
    y_grid = np.linspace(y.min(), y.max(), 200)

    # plot colored clusters
    cluster_idx = 0
    for lbl in unique_labels:
        mask = labels == lbl
        xi,yi = x[mask], y[mask]

        if lbl == -1:
            color,name = 'gray', f'Noise (n={mask.sum()})'
        else:
            cmap_name = (cmaps or default_cmaps)[cluster_idx % len(cmaps or default_cmaps)]
            color = plt.get_cmap(cmap_name)(0.7)
            name = f'Cluster {lbl} (n={mask.sum()})'
            cluster_idx += 1

        if len(xi) > max_points_per_cluster:
            idxs = np.random.choice(len(xi), max_points_per_cluster, replace=False)
            xi_s, yi_s = xi[idxs], yi[idxs]
        else:
            xi_s, yi_s = xi,yi

        ax_main.scatter(xi_s,yi_s, color=color, alpha=alpha, s=s, edgecolors='None', rasterized=True,
                        label=name)

        if len(xi_s) >= 10:
            dens = gaussian_kde(xi_s)(x_grid)
            ax_top.plot(x_grid, dens, color=color, label=name)
            ax_top.fill_between(x_grid, dens, color=color, alpha=0.3)

        if len(yi_s) >= 10:
            dens = gaussian_kde(yi_s)(y_grid)
            ax_right.plot(dens,y_grid,color=color)
            ax_right.fill_betweenx(y_grid, dens, color=color, alpha=0.3)

    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.legend(fontsize=7, loc='best')

    ax_top.set_title(title)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.set_ylabel('Density')

    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.set_xlabel('Density')

    pdf.savefig(fig)
    plt.close(fig)

def plot_k_distance(pdf, min_samples, x_sample, trim_pct = None, include_table = False, feature_names = None, raw_score_sample = None):

    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(x_sample)
    distances, _ = nbrs.kneighbors(x_sample)
    raw_k_distances = distances[:,-1]

    # plot worst points information if specified
    if include_table:

        # get data
        worst_idx = np.argsort(raw_k_distances)[-5:]
        worst_vals = x_sample[worst_idx]
        worst_dist = raw_k_distances[worst_idx]
        worst_raw_scores = raw_score_sample[worst_idx]

        # plot point info to a table
        table_data = [
            [f"#{rank+1}"] + [f"{v:.4g}" for v in worst_vals[rank]] + 
            [f"{worst_dist[rank]:.4g}", f"{worst_raw_scores[rank]:.4g}"]
            for rank in range(len(worst_idx))
        ]
        col_labels = ['Rank'] + feature_names + ['K-Distance', 'Raw CWT Score']
        plot_table(pdf, 'Most Extreme K-Distance Points', table_data, col_labels)

    # get sorted k distances and trim trin_pct highest valeus
    k_distances = np.sort(raw_k_distances)
    if trim_pct is not None:
        trimmed_val =np.percentile(k_distances, trim_pct)
        trimmed = k_distances[k_distances <= trimmed_val]
    else:
        trimmed = k_distances

    # find elbow point to plot
    elbow_idx, eps_estimate = find_elbow(trimmed)

    fig,ax = plt.subplots()
    ax.plot(trimmed)
    ax.scatter(elbow_idx, eps_estimate, color='red', s=8, label=f"Suggested eps = {eps_estimate:.3f}")
    ax.set_xlabel('Points sorted by distance')
    ax.set_ylabel(f'Distance to {min_samples}th nearest neighhor')
    ax.set_title("k-distance plot for eps selection")
    ax.legend()

    pdf.savefig(fig)
    plt.close(fig)

def find_elbow(k_distances):
    y = np.asarray(k_distances)
    x = np.arange(len(k_distances))

    # normalize axes
    y_norm = (y-y.min()) / (y.max() - y.min())
    x_norm = (x-x.min()) / (x.max() - x.min())

    # find elbow
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = (p2-p1)
    line_vec /= np.linalg.norm(line_vec)

    points = np.column_stack([x_norm,y_norm])
    vec_from_p1 = points - p1

    proj_length = vec_from_p1 @ line_vec
    proj_point = np.outer(proj_length, line_vec)
    perp_dist = np.linalg.norm(vec_from_p1 - proj_point, axis=1)

    elbow_idx = np.argmax(perp_dist)

    return elbow_idx, y[elbow_idx]

def plot_table(pdf, title, data, collabels):

    ncols = len(collabels)
    fig_width = max(8, ncols*1.5)
    fig_height = max(8,len(data)*1.5)

    fig,ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.05)
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
    table.auto_set_column_width(col=list(range(len(collabels))))

    ax.set_title(title)

    fig.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)

def plot_skree(pdf, explained_variance_ratios):

    fig,ax = plt.subplots()

    components = np.arange(1, len(explained_variance_ratios)+1)

    ax.bar(components, explained_variance_ratios, label='Per-Component Variance')
    ax.plot(components, np.cumsum(explained_variance_ratios), color='red', marker='o', label='Cumulative Variance')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Varaince Ratio')
    ax.set_title('PCA Skree Plot')
    ax.set_xticks(components)
    ax.legend()

    pdf.savefig(fig)
    plt.close(fig)

def rolling_median_2d(arr, window=51):
    pad = window//2
    padded = np.pad(arr, ((0,0), (pad,pad)), mode='edge')
    windows = sliding_window_view(padded, window_shape=window, axis=1)
    return np.nanmedian(windows,axis=-1)

def quantile_normalization(arr, output_distribution='normal', n_quantiles=1000):
    result = np.empty_like(arr, dtype=float)
    for i in range(arr.shape[0]):
        n_quantiles = min(n_quantiles, arr.shape[1])
        qt = QuantileTransformer(output_distribution=output_distribution, n_quantiles=n_quantiles)
        result[i] = qt.fit_transform(arr[i].reshape(-1,1)).ravel()
    return result

def normalize_matrix(arr, axis=1, eps=1e-9, threshold=1e-6, norm_method='modz'):

    med = np.nanmedian(arr, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(arr-med), axis=axis, keepdims=True) * 1.4826

    global_fallback = np.nanmedian(mad[mad>threshold])
    scale = np.where(mad>threshold, mad, global_fallback)

    modz_matrix = (arr-med) / (scale+eps)

    # add additional compression to bring outlier scale to a more reasonable value
    if norm_method == 'compressed':
        return np.arcsinh(modz_matrix)
    
    return modz_matrix

def subsampled_pca(X, n_components=None, sample_size=20_000):

    n_scans = X.shape[0]
    idx = np.random.choice(n_scans, size=min(sample_size, n_scans), replace=False)
    X_sample = X[idx]

    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_sample)
    explained_variance_ratio = pca.explained_variance_ratio_

    return scores, explained_variance_ratio, idx

def adaptive_linthresh(values, fallback=1e-9):
    values = np.asarray(values)
    nonzero = values[values != 0]
    return np.abs(nonzero).min() if len(nonzero) > 0 else fallback

def plot_chromatogram(pdf, values, labels, ion, alpha = 0.8):
    """
    plots a chromatogram with sections colored according to supplied labels
    """

    y = np.asarray(values)
    x = np.arange(len(y))
    labels = np.asarray(labels)

    cmap = plt.get_cmap('tab10')

    fig,ax = plt.subplots(figsize=(12,6))

    cluster_idx = 0
    for lbl in np.unique(labels):
        mask = labels == lbl
        if lbl == -1:
            ax.scatter(x[mask], y[mask], color='lightgrey', s=5, alpha=alpha/2,
                        label=f'Noise (n={mask.sum()})', rasterized=True)
        else:
            ax.scatter(x[mask], y[mask], color=cmap(cluster_idx%10), s=5, alpha=alpha,
                       label=f'Cluster (n={mask.sum()})', rasterized=True)
            cluster_idx += 1

    ax.set_xlabel("Scan")
    ax.set_ylabel("Intensity")
    ax.set_title(f"TIC for Ion = {ion}")
    ax.legend(markerscale=3, fontsize=8, loc='best')

    pdf.savefig(fig)
    plt.close(fig)

def plot_gs_heatmaps(pdf, gs_params, results):

    # sort params
    sorted_params = {name: sorted(vals) for name, vals in gs_params.items()}

    # define metrics
    metrics = ['validity', 'noise_frac']

    # get param oprder for key consistency
    param_order = list(sorted_params.keys())
 
    # get lookup for full max row
    max_lookup = {(mcs, ms, cse): (mcs, ms, cse, n_clusters, noise_frac, validity) 
                  for mcs, ms, cse, n_clusters, noise_frac, validity in results}   
    # total possible combinations list
    plot_combos = list(combinations(sorted_params.keys(), 2))

    # store maximum values
    maxes = []

    for metric in metrics:

        # get the metric we are coloring heatmap by
        if metric == 'validity':
            lookup = {(mcs, ms, cse): validity for mcs, ms, cse, n_clusters, noise_frac, validity in results}
        elif metric == 'noise_frac':
            lookup = {(mcs, ms, cse): noise_frac for mcs, ms, cse, n_clusters, noise_frac, validity in results}
        else:
            raise ValueError('Unknown metric specified')
        
        # generate heatmaps
        for param1, param2 in plot_combos:

            # get the index for the other param, get the number of values for that param and map this ij to that
            k = [name for name in sorted_params.keys() if name not in (param1, param2)][0]
            n_facets = len(sorted_params[k])

            # facet by eps because it is a 'secondary' metric
            if k != 'eps':
                continue

            # calculate number of rows/cols for faceted heatmap display
            n_facet_rows = int(np.ceil(np.sqrt(n_facets)))
            n_facet_cols = int(np.ceil(n_facets / n_facet_rows))

            fig,axes = plt.subplots(n_facet_rows, n_facet_cols, squeeze=False,
                                    figsize=(6*n_facet_cols, 5*n_facet_rows))

            # generate faceted heatmaps
            for idx, facet_val in enumerate(sorted_params[k]):

                row, col = divmod(idx, n_facet_cols)
                ax = axes[row,col]

                grid = np.full((len(sorted_params[param1]),len(sorted_params[param2])), np.nan)
                for a, val_1 in enumerate(sorted_params[param1]):
                    for b, val_2 in enumerate(sorted_params[param2]):
                        values_by_name = {param1:val_1, param2:val_2, k:facet_val}
                        lookup_key = tuple(values_by_name[name] for name in param_order)
                        grid[a,b] = lookup.get(lookup_key, np.nan)

                im = ax.imshow(grid, cmap='viridis', aspect='auto')

                ax.set_xticks(range(len(sorted_params[param2]))); ax.set_xticklabels(sorted_params[param2], rotation=45)
                ax.set_yticks(range(len(sorted_params[param1]))); ax.set_yticklabels(sorted_params[param1])

                ax.set_xlabel(param2); ax.set_ylabel(param1)

                fig.colorbar(im, ax=ax, shrink=0.7)

                # find the maximum value and add it to maxes
                if metric == 'validity':
                    i,j = np.unravel_index(np.nanargmax(grid), grid.shape)
                else:
                    i,j = np.unravel_index(np.nanargmin(grid), grid.shape)
                max_vals_by_name = {
                    param1:sorted_params[param1][i],
                    param2:sorted_params[param2][j],
                    k:facet_val}
                max_key = tuple(max_vals_by_name[name] for name in param_order)
                max_vals = max_lookup[max_key]
                mcs, ms, cse, n_clusters, noise_frac, validity = max_vals
                max_row = [metric, k, mcs, ms, cse, n_clusters, round(noise_frac, 3), round(validity, 3)]
                maxes.append(max_row)
                    
            # hide any unused panels
            for idx in range(n_facets, n_facet_rows * n_facet_cols):
                row,col = divmod(idx, n_facet_cols)
                axes[row,col].axis('off')

            fig.suptitle(f"{metric}: {param2} vs {param1} (faceted by {k})", fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    # plot a table of maximum combinations
    results_labels = ['maximized_metric', 'faceted_prameter', 'min_cluster_size', 'min_samples', 'eps', 'n_clusters', 'noise_fraction', 'validity']
    table_data = [list(row) for row in maxes]
    plot_table(pdf, f"Best Combinations per-facet per-metric", table_data, results_labels)

def plot_hdbscan_summary(pdf, results, top_n=3, n_clusters_max=5):
    """
    plots summary statistics for a given gridsearch of a given 
    """

    # pull out data from results
    results_matrix = np.array(results)
    results_matrix = results_matrix[~np.isnan(results_matrix[:6])]
    if len(results_matrix) == 0:
        return

    seeds = results_matrix[:,0]
    mcs_vals = results_matrix[:,1]
    ms_vals = results_matrix[:,2]
    cse_vals = results_matrix[:,3]
    n_cluster_vals = results_matrix[:,4]
    noise_frac_vals = results_matrix[:,5]
    validity_vals = results_matrix[:,6]

    # plot metric histograms
    plot_histogram(pdf, 'n_clusters distribtution', 'n_clusters', n_cluster_vals)
    plot_histogram(pdf, 'noise_frac distribution', 'noise_frac', noise_frac_vals)
    plot_histogram(pdf, 'validity distribution', 'validity', validity_vals)

    # plot parameter combination histograms
    cluster_mask = n_cluster_vals < n_clusters_max
    mcs_f = mcs_vals[cluster_mask]
    ms_f = ms_vals[cluster_mask]
    cse_f = cse_vals[cluster_mask]
    validity_f = validity_vals[cluster_mask]
    seeds_f = seeds[cluster_mask]



