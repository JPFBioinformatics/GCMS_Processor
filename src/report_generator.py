# region Imports

import sys
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# location of pipeline root dir
root_dir = Path(__file__).resolve().parent.parent
# tell python to look here for modules
sys.path.insert(0, str(root_dir))

from src.config_loader import ConfigLoader

# endregion

class ReportGenerator:
    """
    Class that will generate a report based on a list of identified peaks
    """

    def __init__(self, cfg: ConfigLoader, peaks: dict):
        """
        Params:
            cfg                     loaded config for this run
            peaks                   dict of sample_name: peak list for all samples
        """
        self.peaks = peaks
        self.cfg = cfg
        self.data = None
        self.groups = None

    def generate_report(self, sym_bin: float = 0.1, rt_bin: float = 0.01, num_pcs: int = 5):
        """
        Generates a single pdf report for the entire run
        """
        cfg = self.cfg
        input_dir = cfg.get("input_dir")
        name = cfg.get("run_name")
        out_file = Path(input_dir) / f"{name}_report.pdf"

        with PdfPages(out_file) as pdf:
            self.qc_plots(pdf=pdf,sym_bin=sym_bin,rt_bin=rt_bin)
            self.calculate_pca(pdf=pdf,num_comps=num_pcs)

    def assign_group(self, case: list, control: list):
        """
        Assigns samples to case/control group (case = 1 control = 0)
        Params:
            case/control                    lists of the sample names for samples in case and control group respectively
        """

        # get sample map and start empty groups list
        samples = self.sample_map
        groups = []

        # add each sample designation to the correct place
        for name,idx in samples.items():
            if name in control:
                groups[idx] = 0
            elif name in case:
                groups[idx] = 1
            else:
                raise ValueError(f"Sample not found in case/control lists")
            
        self.groups = groups

    def qc_plots(self, pdf, sym_bin: float = 0.1, rt_bin: float = 0.01):
        """
        Generates a pdf page that contains 4 plots, a bound symmetry and a rt difference histogram as well as box and whisker plots for both
        Params:
            pdf                                     pdf file to save figure to
            sym_bin,rt_bin                          bin size for the symmetry histogram and for the retention time histogram
        """
        cfg = self.cfg
        sym_threshold = cfg.get("endpoint_threshold")
        rt_threshold = cfg.get("rt_threshold")

        # grab rt diff and symmetry values
        rt_diffs = []
        symmetry_values = []
        for peak in self.peaks:
            rt_diffs.append(peak["rt_diff"])
            symmetry_values.append(peak["bound_symmetry"])

        if not rt_diffs or not symmetry_values:
            raise ValueError("No peak data availabe for QC plots")

        # find maxes and mins for binning
        sym_min = min(symmetry_values)
        sym_max = max(symmetry_values)
        rt_min = min(rt_diffs)
        rt_max = max(rt_diffs)

        # bin
        sym_bins = np.arange(
            sym_min,
            sym_max + sym_bin,
            sym_bin
        )
        rt_bins = np.arange(
            rt_min,
            rt_max + rt_bin,
            rt_bin
        )

        # genreate figures 
        fig,axes = plt.subplots(2,2,figsize=(11,8.5))
        fig.suptitle("QC Metrics Summary",fontsize=14)
        fig.tight_layout(rect=[0,0,1,0.95])

        # symmetry histogram
        axes[0,0].hist(symmetry_values,bins=sym_bins)
        axes[0,0].set_xlim(sym_min,sym_max)
        axes[0,0].set_title("Bound Symmetry Distribution")
        axes[0,0].set_xlabel("Bound Symmetry")
        axes[0,0].set_ylable("Count")
        axes[0,0].axvline(
            sym_threshold,
            linestyle="--",
            linewidth=2,
            label="Symmetry Threshold"
        )
        axes[0,0].legend()

        # symmetry box and whisker plot
        axes[0,1].boxplot(symmetry_values,vert=True,showfliers=True)
        axes[0,1].set_title("Bound Symmetry Plot")
        axes[0,1].set_ylabel("Symmetry Value")
        axes[0,1].legend()

        # rt difference histogram
        axes[1,0].hist(rt_diffs,bins=rt_bins)
        axes[1,0].set_xlim(rt_min,rt_max)
        axes[1,0].set_title("RT Difference Distribution")
        axes[1,0].set_xlabel("RT Difference (min)")
        axes[1,0].set_ylabel("Count")
        axes[1,0].axvline(
            rt_threshold,
            linestyle="--",
            linewidth=2,
            label="RT Threshold"
        )
        axes[1,0].legend()

        # rt difference box and whisker plot
        axes[1,1].boxplot(rt_diffs,vert=True,showfliers=True)
        axes[1,1].set_title("RT Difference Plot")
        axes[1,1].set_ylabel("RT Difference")
        axes[1,1].legend()

        # save figure
        pdf.savefig(fig)
        plt.close(fig)
        
    def generate_matrix(self):
        """
        Takes abundance/area data and stores it in a matrix as well as generating maps for samples and molecules
        matrix is organized with rows = samples and columns = molecules
        method will also reformat peaks dict so that it is simply sample_name: peak area list for each peak
        the molecule and sample maps are used to decode this
        """
        # get peak data and generate maps
        peaks = self.peaks

        sample_names = list(peaks.keys())
        num_samples = len(sample_names)
        sample_map = {}
        for idx,name in enumerate(sample_names):
            sample_map[name] = idx
        
        num_peaks = len(peaks[sample_names[0]])
        molecule_map = {}
        for idx,peak in enumerate(peaks[sample_names[0]]):
            molecule_map[peak["molecule"]] = idx

        # save peak data in a 2d numpy array
        data = np.zeros((num_samples,num_peaks))

        for i,sample in enumerate(sample_names):
            for j,peak in enumerate(peaks[sample]):
                if peak["rt_valid"]:
                    data[i,j] = peak['area']
                else:
                    data[i,j] = 0

        # save data to object
        self.matrix = data
        self.sample_map = sample_map
        self.molecule_map = molecule_map

    def write_to_excel(self):
        """
        Writes peak area values to an excel table for manual analysis
        """
        cfg = self.cfg
        input_dir = cfg.get("input_dir")
        out_file = Path(input_dir) / "data.xlsx"

        # sort sample and molecule maps to ensure correct ordering
        samples_ordered = [sample for sample, idx in sorted(self.sample_map.items(), key=lambda x: x[1])]
        molecules_ordered = [mol for mol, idx in sorted(self.molecule_map.items(), key=lambda x: x[1])]

        # place data in dataframe
        df = pd.DataFrame(self.matrix, index=samples_ordered, columns=molecules_ordered)

        # write to excel
        df.to_excel(out_file,index=True)

    def calculate_pca(self, pdf, num_comps: int = 2):
        """
        Generates a PCA plot for the data stored in this object (self.matrix) and saves output to report pdf
        Params:
            pdf                             pdf file to save figure to
            num_comps                       the number of PCA components to include in plot
        """

        # grab data matrix and threshold values
        data = self.matrix
        cfg = self.cfg
        var_threshold = cfg.get("variance_threshold")
        pca_var = cfg.get("pca_var")

        if data == None:
            raise ValueError("No data matrix found for PCA plotting")

        # convert to ppm
        data = (data / data.sum(axis=1,keepdims=True)) * 1e6

        # log transform
        data = np.log1p(data)

        # filter out low varaince features
        feature_var = data.var(axis=0)
        keep = feature_var > var_threshold
        data = data[:,keep]

        # z-score transform
        data = StandardScaler().fit_transform(data)

        # calculate PCA and explained variance
        pca = PCA(n_components=num_comps)
        scores = pca.fit_transform(data)
        variance = pca.explained_variance_ratio_

        # plot variance bar graph
        fig,ax=plt.subplots(figsize=(6,4))
        fig.tight_layout()

        ax.bar(range(1,num_comps+1), variance, color="skyblue", edgecolor = 'k')
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("PCA explained Variance")
        ax.set_xticks(range(1,num_comps+1))
        ax.legend()

        # save figure
        pdf.savefig(fig)
        plt.close(fig)

        # determine how many pcs to plot (those that sum to account for 80% of varaince by default)
        cumulative_var = np.cumsum(variance)
        num_to_plot = np.searchsorted(cumulative_var, pca_var) + 1

        # plot relevant pca combinations
        for i in range(num_to_plot - 1):
            for j in range(i+1,num_to_plot):
                pc_pair = scores[:,[i,j]]
                self.plot_pca(pc_pair,num_pcx=i+1, num_pcy=j+1, pdf=pdf, color_groups=self.groups)

    def plot_pca(self, pc_scores, num_pcx: int, num_pcy: int, pdf, color_groups: list = None, color_molecule: str = None):
        """
        Generates a PCA plot for a given two principal components, helper method for calculate_pca
        Params:
            pc_scores                       2d numpy array with shape (num_samples,2) that contains your two pcs to plot
            num_pcx/num_pcy                 numbers of the pcs being graphed (for labels)
            pdf                             pdf file to save figures to
            color_groups                    list of int values for how many groups you want to color bay (1 = case, 0 control for example)
            color_molecule                  name of a molecule that you want to color by abundance of
        """

        fig,ax = plt.subplots(figsize=(6,6))
        fig.tight_layout()

        # color samples based on gruoup (explicit, not continuous)
        if color_groups:

            # seperate up to 10 groups with unique colors for each
            unique_groups = list(set(color_groups))
            colors = plt.cm.tab10.colors
            group_color_map = {g: colors[i % 10] for i, g in enumerate(unique_groups)}

            # add color to each point and plot
            for g in unique_groups:
                for g in unique_groups:
                    idx = [i for i, grp in enumerate(color_groups) if grp == g]
                    ax.scatter(pc_scores[idx, 0], pc_scores[idx, 1],
                        label=g, color=group_color_map[g], alpha=0.8)
                    
            ax.set_xlabel(f"PCA {num_pcx}")
            ax.set_ylabel(f"PCA {num_pcy}")
            ax.set_title(f"{num_pcy} vs {num_pcx}")
            ax.legend()

        # color samples based on a specifed value
        elif color_molecule:

            # get column from matrix that corrosponds to the molecule to color by
            col_idx = self.molecule_map[color_molecule]
            values = self.matrix[:, col_idx]
            
            # normalize and generate color maps
            norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["blue","red"])

            # plot
            sc = ax.scatter(pc_scores[:,0], pc_scores[:,1], c=values, cmap=cmap, norm=norm, alpha=0.8)
            fig.colorbar(sc).set_label(f"{color_molecule} Abundance")
            ax.set_xlabel(f"PC {num_pcx}")
            ax.set_ylabel(f"PC {num_pcy}")
            ax.set_title(f"{num_pcy} vs {num_pcx}")
            ax.legend()
        
        # do not color samples if not specified
        else:
            ax.scatter(pc_scores[:,0], pc_scores[:,1], alpha=0.8)
            ax.set_title(f"{num_pcy} vs {num_pcx}")
            ax.set_xlabel(f"PC {num_pcx}")
            ax.set_ylabel(f"PC {num_pcy}")
            

        # save figure
        pdf.savefig(fig)
        plt.close(fig)