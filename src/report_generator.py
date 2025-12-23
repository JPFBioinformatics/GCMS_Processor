# region Imports

import sys
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from openpyxl import load_workbook
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

    def __init__(self, cfg: ConfigLoader, peaks: dict = None):
        """
        Params:
            cfg                     loaded config for this run
            peaks                   dict of peak values for each sample (sample: list of peak dicts)
        """
        self.cfg = cfg
        self.peaks = peaks
        self.matrix = None
        self.data = None
        self.groups = None
        self.value_map = None
        self.standards = None
        self.sample_map = None
        self.norm_matrix = None

    def save_peaks(self, peaks: dict):
        """
        Saves a peak dict as self.peaks ifd not specified when object is creatd
        Params:
            peaks                   dict of sample: peak list where each peak is a dict for the full set of samples for data analysis
        """
        self.peaks = peaks

    def normalize_matrix(self):
        """
        Normalizes the values matrix to given standards/normalization factors as well as one hot encodes grouping
        """

        if self.matrix is None:
            raise ValueError("Must have generated matrix before loading data")

        # get a copy of matrix/value map
        norm_matrix = self.matrix.copy()
        norm_value_map = self.value_map.copy()

        # get file paths
        cfg = self.cfg
        template_name = cfg.get("template_file")
        file = Path(cfg.get("input_dir")) / template_name

        # read file (without header) and load moleucle, mz values, and retention times to lists
        df = pd.read_excel(file,skiprows=3)

        samples = df["samples"].dropna().to_list()
        groups = df["group"].dropna().to_list()
        norm_factors = df["norm"].dropna().to_list()
        self.norm = norm_factors
        standards = df["standard"].dropna().to_list()

        # onehot encode groups and save them to matrix, updating value_map
        if groups:
            self.num_groups = len(groups)
            group_onehot = pd.get_dummies(groups).to_numpy()
            norm_matrix = np.hstack([norm_matrix,group_onehot])
            num_values = len(norm_value_map)
            for i,col_name in enumerate(pd.get_dummies(groups).columns):
                norm_value_map[num_values+i] = col_name
        
        # adjust samples to intenal standard
        norm_standards = {}
        if standards:
            standard_idxs = []
            for i,std_name in enumerate(standards):
                if std_name not in self.value_map:
                    print(self.value_map)
                    raise ValueError(f"Standard {std_name} not found in collected values")
                std_idx = norm_value_map[std_name]
                norm_standards[std_name] = norm_matrix[:,std_idx].copy()
                standard_idxs.append(std_idx)
                norm_matrix[:,i] = norm_matrix[:,i] / norm_matrix[:,std_idx]

            # remove standard rows from matrix/value map and rebuild map to corrospond to new matrix
            norm_matrix = np.delete(norm_matrix, standard_idxs, axis=1)
            for idx in sorted(standard_idxs, reverse=True):
                norm_value_map.pop(idx,None)
            norm_value_map = {name:i for i,name in enumerate(norm_value_map.values())}

        # adjust data to normalization factor
        for j, sample in enumerate(samples):
            if sample not in self.sample_map:
                raise ValueError(f"Sample {sample} not found in collected samples")
            norm_factor = norm_factors[j]
            matrix_idx = self.sample_map[sample]
            norm_matrix[matrix_idx,:] = norm_matrix[matrix_idx,:] / norm_factor

        # save normalized data
        self.norm_matrix = norm_matrix
        self.norm_value_map = norm_value_map
        self.norm_standards = norm_standards
            
    def generate_template(self):
        """
        generates a templeate xlsx file for iputting m/z and rt values
        """
        # get input dir
        cfg = self.cfg
        input_dir = Path(cfg.get("input_dir"))
        out_dir = Path(cfg.get_path("results_dir"), input_dir)
        template = cfg.get("template_file")
        file = out_dir / template

        # get list of sample names from input dir
        names = sorted(
            p.stem
            for p in input_dir.iterdir()
            if p.is_dir() and p.suffix == ".D"
        )

        # generate sample table df
        sample_df = pd.DataFrame({
            "samples": names,
            "group": ['' for _ in names],
            "norm": ['' for _ in names],
        })

        # generate data df 
        data_df = pd.DataFrame(columns=["molecule","mz","rt","standard"])

        # generate headers
        header1 = "Template file for gcms automatic peak picking/integration, please ONLY fill in appropriate values and feel free to leave case/control empty if need be"
        header2 = "group = grouping for samples (ie case/control), norm = normalization factor, molecule = id of this moleucue, mz = ion to measure, rt = peak retention time, standard = name of standard to apply to that sample"

        # add sample/data dfs to excel file
        with pd.ExcelWriter(file, engine="openpyxl") as writer:

            sample_df.to_excel(
                writer,
                index=False,
                startrow=3,
                startcol=1
            )

            data_df.to_excel(
                writer,
                index=False,
                startrow=3,
                startcol=5
            )
        
        # add headers
        wb = load_workbook(file)
        ws = wb.active

        ws["A1"] = header1
        ws["A2"] = header2

        wb.save(file)

    def generate_report(self, sym_bin: float = 0.1, rt_bin: float = 0.01, num_pcs: int = 5):
        """
        Generates a single pdf report for the entire run
        """
        cfg = self.cfg
        input_dir = Path(cfg.get("input_dir"))
        out_dir = cfg.get_path("results_dir", input_dir)
        name = cfg.get("run_name")
        out_file = Path(out_dir) / f"{name}_report.pdf"

        with PdfPages(out_file) as pdf:
            self.std_qc_plots(pdf=pdf)
            self.qc_plots(pdf=pdf,sym_bin=sym_bin,rt_bin=rt_bin)
            self.pca_plots(pdf=pdf,num_comps=num_pcs)

    def std_qc_plots(self, pdf):
        """
        Generates a box and whisker polot for standard values, naming samples that are outliers in standard amount
        Params:
            pdf                         pdf file to save figure to
        """
        # get name and number of standards
        std_names = list(self.norm_standards.keys())
        num_stds = len(std_names)

        # generate figure
        fig,axes = plt.subplots(1,num_stds,figsize=(num_stds*2,6), squeeze=False)

        # generate inverse sample map
        inv_smaple_map = {v:k for k,v in self.sample_map.items()}

        for i,std_name in enumerate(std_names):

            # geet values for this std
            values = self.norm_standards[std_name]

            # plot
            ax = axes[0,i]

            # calculate values for outlier detection
            q1,q3 = np.percentile(values,[25,75])
            iqr = q3-q1
            lower = q1- 1.5 * iqr
            upper = q3 + 1.5 * iqr

            for j,val in enumerate(values):
                if val < lower or val > upper:
                    ax.text(1,val,inv_smaple_map[j], fontsize=8, rotation=45, ha="right")

            ax.set_title(f"Standard {std_name}")
            ax.set_ylabel("Abundance")
            ax.set_xticks([])

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def qc_plots(self, pdf, sym_bin: float = 0.1, rt_bin: float = 0.01, width_bin: float = 1.0):
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
        flat_top_flags = []
        width_flags = []
        widths = []
        rt_diffs = []
        symmetry_values = []
        for peak in self.peaks:
            flat_top_flags.append(peak["flat_top"])
            width_flags.append(peak["width_flag"])
            widths.append(peak["right_bound"] - peak["left_bound"])
            rt_diffs.append(peak["rt_diff"])
            symmetry_values.append(peak["bound_symmetry"])

        if not rt_diffs or not symmetry_values:
            raise ValueError("No peak data availabe for QC plots")

        # calcualte flat top and width flag percents
        percent_ft = 100 * sum(flat_top_flags) / len(flat_top_flags)
        width_counts = Counter(width_flags)
        num_peaks = len(width_flags)
        percent_width = {k: 100*v/num_peaks for k,v in width_counts.items()}
        
        # generate summary text
        summary_text = f"Flat-Top Peak Occurance: {percent_ft}%\n\nWidth Catagory Occurance:\n"
        for cat,pct in percent_width.items():
            summary_text += f"\t{cat}: {pct:.1f}%/n"

        # find maxes and mins for binning
        width_min = min(widths)
        width_max = max(widths)
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
        width_bins = np.arange(
            width_min,
            width_max + width_bin,
            width_bin
        )

        # genreate figures 
        fig,axes = plt.subplots(3,2,figsize=(11,8.5))
        fig.suptitle("QC Metrics Summary",fontsize=14)
        fig.tight_layout(rect=[0,0,1,0.95])

        # summary text
        fig.text(
            0.05, 0.95,
            summary_text,
            fontsize = 10,
            va = 'top',
            ha = 'left',
            bbox=dict(facecolor = "white", alpha = 0.5)
        )

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
        axes[1,1].boxplot(widths,vert=True,showfliers=True)
        axes[1,1].set_title("RT Difference Plot")
        axes[1,1].set_ylabel("RT Difference")
        axes[1,1].legend()

        # width histogram
        axes[2,0].hist(rt_diffs,bins=width_bins)
        axes[2,0].set_xlim(rt_min,rt_max)
        axes[2,0].set_title("Width Distribution")
        axes[2,0].set_xlabel("Width (scans)")
        axes[2,0].set_ylabel("Count")
        axes[2,0].axvline(
            25,
            linestyle="--",
            linewidth=2,
            label="Width Threshold"
        )
        axes[2,0].legend()

        # width box and whisker plot
        axes[2,1].boxplot(widths,vert=True,showfliers=True)
        axes[2,1].set_title("Width Plot")
        axes[2,1].set_ylabel("Width")
        axes[2,1].legend()

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
        if self.peaks:
            peaks = self.peaks
        else:
            raise ValueError("No peak dict found")

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
        self.value_map = molecule_map

    def write_to_excel(self, out_type: str = "dual"):
        """
        Writes peak area values to an excel table for manual analysis
        Params:
            type                    "raw" if only want raw values "norm" if you want normalized, "dual" if you want both
        """
        cfg = self.cfg
        input_dir = cfg.get("input_dir")
        results_dir = cfg.get_path("results_dir", input_dir)
        out_file = Path(results_dir) / "results.xlsx"

        # sort sample and molecule maps to ensure correct ordering
        samples_ordered = [sample for sample, idx in sorted(self.sample_map.items(), key=lambda x: x[1])]
        molecules_ordered = [mol for mol, idx in sorted(self.value_map.items(), key=lambda x: x[1])]
        norm_molecules_ordered = [mol for mol,idx in sorted(self.norm_value_map.items(), key=lambda x: x[1])]

        # generate excel file
        with pd.ExcelWriter(out_file,engine="openpyxl") as writer:

            # get raw values
            if out_type == "raw" or out_type == "dual":
                df_raw = pd.DataFrame(self.matrix, index=samples_ordered, columns=molecules_ordered)
                df_raw.to_excel(writer, sheet_name="raw", index=True)

            # get normalized values
            if out_type == "norm" or out_type == "dual":
                if not self.norm_matrix:
                    raise ValueError(f"Normailze matrix before returning normalized data")
                
                df_norm = pd.DataFrame(self.norm_matrix, index=samples_ordered, columns=norm_molecules_ordered)
                df_norm.to_excel(writer, sheet_name="normalized", index=True)

    def pca_plots(self, pdf, num_comps: int = 2):
        """
        Generates a PCA plot for the data stored in this object (self.matrix) and saves output to report pdf
        Params:
            pdf                             pdf file to save figure to
            num_comps                       the number of PCA components to include in plot
        """
        # get matrix and remove the one-hot encoded groups if needed
        if self.num_groups > 0:
            group_data = self.data[:, :-self.num_groups:]
            data = self.data[:,:-self.num_groups]
        else:
            group_data = None
            data = self.data

        # grab threshold values
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
        if group_data is not None:
            data = np.hstack([data,group_data])
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

    def plot_pca(self, pc_scores, num_pcx: int, num_pcy: int, pdf, color_groups: bool = True, color_molecule: str = None):
        """
        Generates a PCA plot for a given two principal components, helper method for calculate_pca
        Params:
            pc_scores                       2d numpy array with shape (num_samples,2) that contains your two pcs to plot
            num_pcx/num_pcy                 numbers of the pcs being graphed (for labels)
            pdf                             pdf file to save figures to
            color_groups                    bool True if you want to color by group false if you don't want to
            color_molecule                  name of a molecule that you want to color by abundance of
        """

        fig,ax = plt.subplots(figsize=(6,6))
        fig.tight_layout()

        # color samples based on gruoup (explicit, not continuous)
        if color_groups:

            # get group cols
            all_keys = self.value_map.keys()
            group_keys = all_keys[-self.num_groups]
            group_onehot_cols = self.matrix[:,-self.num_groups:]
            color_groups = np.argmax(group_onehot_cols, axis=1)

            # map index to group name for legend
            index_to_group = {i:name for i,name in enumerate(group_keys)}

            # assign colors to groups
            unique_groups = sorted(set(color_groups))
            colors = plt.cm.tab10.colors
            group_color_map = {g:colors[i%10] for i,g in enumerate(unique_groups)}

            # add color to each point and plot
            for g in unique_groups:
                idx = [i for i,grp in enumerate(color_groups) if grp == g]
                ax.scatter(pc_scores[idx,0], pc_scores[idx,1], label = index_to_group[g], color = group_color_map[g], alpha=0.8)
                    
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

    def stack_new_col(self, values: np.ndarray, label: str):
        """
        Method to add a new column to the data array, used to insert case/control, sex etc.. labels
        Params:
            values                      valid list to add as a new column
            label                       label to place new column under in value map
        """
        # ensure values and matrix are compatible size
        matrix = self.matrix
        num_rows = matrix.shape[0]
        if len(values) != num_rows:
            raise ValueError(f"Incorrect column lenght, {len(values)} rows cannot be added to a matrix with {num_rows} rows")
        
        # if compatible then continue
        new_col = np.ndarray(values).reshape(-1,1)
        matrix = np.hstack((matrix,new_col))

        # add to value map
        self.value_map[label] = num_rows

        # save new matrix
        self.matrix = matrix