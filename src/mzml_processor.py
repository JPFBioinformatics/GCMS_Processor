# region Imports

import subprocess, base64, zlib, sys
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

# location of pipeline root dir
root_dir = Path(__file__).resolve().parent.parent
# tell python to look here for modules
sys.path.insert(0, str(root_dir))

from src.intensity_matrix import IntensityMatrix
from src.config_loader import ConfigLoader
from src.utils import log_subprocess

# endregion



class MzMLProcessor:
    """
    Processes .d files from agilant GCMS and converts data to .mzML file for downstream processing
    """

    def __init__(self, cfg: ConfigLoader):
        self.cfg = cfg

    def full_bulk_convert(self):
        """
        Converts all compatible .d files in a directory to .mzml files and saves them to a directory in the input directory
        Returns:
            mzml_dir                        location of mzml dir
            matrices                        list of intensitymatrix objects created from all files in mzml
        """
        # load config
        cfg = self.cfg
        input_dir = Path(cfg.get("input_dir"))
        results = Path(cfg.get("results_dir"))
        mzml_dir = input_dir / results / cfg.get("mzml_dir")

        # check to see if mzml files are already converted
        mzml_files = list(mzml_dir.glob("*.mzML"))
        samples_in = list(input_dir.glob("*.D"))

        # generate list of input file names
        in_names = []
        for in_file in samples_in:
            in_names.append(in_file.stem)
        
        # assume input files have already been processed
        process = False

        # if mzml files exist make sure every input file has a corrosponding mzml
        if mzml_files:
            for file in mzml_files:
                mzml_name = file.stem
                if mzml_name not in in_names:
                    process = True
        else:
            process = True

        if process:
            # ensure mzml dir exists
            mzml_dir.mkdir(parents=True,exist_ok=True)

            # get log_dir
            log_dir = input_dir / results

            # run each msconvert and log results
            for sample in samples_in:
                sample_name = str(sample.stem)
                if sample.is_dir():
                    cmd = [
                        "msconvert",
                        str(sample),
                        "--mzML",
                        "--outdir", str(mzml_dir)
                    ]

                    result = subprocess.run(cmd,check=True,capture_output=True,text=True)

                    log_subprocess(result,log_dir,sample_name)
            
            # generate new list of valid mzml files
            mzml_files = list(mzml_dir.glob("*.mzML"))
    
        # sort mzml files
        mzml_files = sorted(mzml_files, key=lambda f: f.stem)
        matrices = []
        for file in mzml_files:
            matrix = self.create_intensity_matrix(file)
            matrices.append(matrix)

        return matrices

    @staticmethod
    def decode_binary_data(encoded_data, dtype):
        """
        Decodes base64, decompresses zlib, and converts to a NumPy array with an associated m/z and time lists.
        Params:
            encoded_data                base64 data to be decoded
            dtype                       type of data that is converted
        """

        decoded = base64.b64decode(encoded_data)
        decompressed = zlib.decompress(decoded)

        return np.frombuffer(decompressed, dtype=dtype)

    @staticmethod
    def bin_masses(unique_mzs, intensity_matrix):
        """
        Bins masses -0.3 to +0.7 of integer values
        Params:
            unique_mzs                      list of m/z values to bin
            intensity_matrix                intensity matrix that corrosponds to unbinned m/z values for processing
        Returns:
            binned_mzs                      list of binned mz values
            binned_matrix                   intensity matrix that has been binned
        """
        
        # change unique mz list to array
        mz_array = np.asarray(unique_mzs)

        # get bin assignments
        bin_assignments = (mz_array + 0.3).astype(int)

        # get unique bins and inverse
        binned_mzs, inverse = np.unique(bin_assignments, return_inverse = True)
        
        # prepare output matrix (rows = bins cols = time points)
        num_bins = len(binned_mzs)
        _, num_cols = intensity_matrix.shape
        binned_matrix = np.zeros((num_bins,num_cols), dtype = intensity_matrix.dtype)

        # bin masses using inverse to map each origional mz to its bin
        for src_row, bin_row in enumerate(inverse):
            binned_matrix[bin_row] += intensity_matrix[src_row]

        return list(binned_mzs), binned_matrix 

    @staticmethod
    def create_scan_matrix(mzml_path):
        """
        Extracts spectra metadata and builds a matrix where each spectrum is
        represented by a column and each unique m/z is represented by a row from a SCAN file
        Params:
            mzml_path                   path to the mzml file to process
        Returns:
            output_matrix               intensitymatrix object based on input mzml file
        """
        tree = ET.parse(mzml_path)
        root = tree.getroot()

        namespaces = {
            '': 'http://psi.hupo.org/ms/mzml'
        }

        spectra_metadata = {}
        intensity_list = []
        unique_mzs = set()

        # get name of sample
        name = mzml_path.stem

        # Iterate over each <spectrum> element
        for spectrum in root.findall('.//spectrum', namespaces):
            scan_id = spectrum.get('id')
            if scan_id:
                scan_id = scan_id.split('=')[-1]  # Split by '=' and take the second part (which is the number)

            scan_start_time = None
            scan_list = spectrum.find('scanList', namespaces)
            if scan_list is not None:
                scan = scan_list.find('scan', namespaces)
                if scan is not None:
                    scan_start_time = scan.find('.//cvParam[@name="scan start time"]', namespaces)
                    if scan_start_time is not None:
                        scan_start_time = scan_start_time.get('value')

            binary_data_elements = spectrum.findall('./binaryDataArrayList/binaryDataArray/binary', namespaces)

            if len(binary_data_elements) < 2:
                print(f"Skipping spectrum {scan_id} due to missing binary data")
                continue

            # Save metadata for the spectrum in the list
            spectra_metadata[int(scan_id)-1] = float(scan_start_time)

            # Grabs the binary encoded m/z and intensity arrays
            mz_array_encoded = binary_data_elements[0].text
            intensity_array_encoded = binary_data_elements[1].text

            mz_array = MzMLProcessor.decode_binary_data(mz_array_encoded, dtype=np.float64)
            intensity_array = MzMLProcessor.decode_binary_data(intensity_array_encoded, dtype=np.float32)

            # Create a dictionary for each spectrum (m/z -> intensity)
            spectrum_intensity_dict = dict(zip(mz_array, intensity_array))

            # Add the spectrum dictionary to the intensity_list
            intensity_list.append(spectrum_intensity_dict)

            # Add the m/z values to the set of unique m/z values
            unique_mzs.update(mz_array)

        # Convert the set of unique m/z values to a sorted list
        unique_mzs = sorted(unique_mzs)

        # Create a NumPy array for the intensity matrix
        intensity_matrix = np.zeros((len(unique_mzs), len(intensity_list)))

        # Create a map from m/z values to row indices in the intensity matrix
        mz_index_map = {mz: idx for idx, mz in enumerate(unique_mzs)}

        # Iterate over the intensity_list to fill the intensity matrix
        for col_idx, spectrum_intensity_dict in enumerate(intensity_list):
            for mz, intensity in spectrum_intensity_dict.items():
                row_idx = mz_index_map.get(mz)
                if row_idx is not None:
                    intensity_matrix[row_idx, col_idx] = intensity

        # bin the intensity matrix and unique_mzs
        binned_mzs, binned_matrix = MzMLProcessor.bin_masses(unique_mzs, intensity_matrix)

        # add TIC row to end of matrix
        sum_row = np.sum(binned_matrix, axis=0)
        final_matrix = np.vstack((binned_matrix,sum_row))

        # add 9999 value to end of binned_mzs to represent the TIC
        binned_mzs.append(9999)

        # create intensity matrix object
        output_matrix = IntensityMatrix(intensity_matrix=final_matrix,unique_mzs=binned_mzs,spectra_name=name,spectra_metadata=spectra_metadata)

        return output_matrix
    
    @staticmethod
    def create_sim_matrix(mzml_path):
        """
        Extracts spectra metadata and builds a matrix where each spectrum is
        represented by a column and each unique m/z is represented by a row, from a SIM file
        Params:
            mzml_path                   path to the mzml file to process
        Returns:
            output_matrix               intensitymatrix object based on input mzml file
        """
        tree = ET.parse(mzml_path)
        root = tree.getroot()

        namespaces = {
            '': 'http://psi.hupo.org/ms/mzml'
        }

        time_map = {}
        ion_map = {}

        # get name of sample
        file_name = mzml_path.stem

        # generate empty matrix for data storage
        chrom_list = root.find('.//chromatogramList', namespaces)
        num_chroms = int(chrom_list.get('count'))
        first_chrom = chrom_list.find('chromatogram',namespaces)
        num_time_points = int(first_chrom.get('defaultArrayLength'))
        matrix = np.zeros((num_chroms,num_time_points))

        int_count = 0
        time_count = 0

        # iterate over chroms and gather data
        for idx,chrom in enumerate(chrom_list):

            # get ion and add to map
            iso = chrom.find(
                './/precursor/isolationWindow/cvParam[@accession="MS:1000827"]',
                namespaces
            )
            # handle TIC ion value
            if iso is None:
                ion = 9999
                ion_map[ion] = num_chroms-1
            else:
                ion = int(0.3+float(iso.attrib["value"]))
                ion_map[ion] = int(idx)-1

            # now parse the binary data arrays, getting the time and intensity arrays
            for bda in chrom.findall('.//binaryDataArray', namespaces):
                cvparams = [child for child in list(bda) if child.tag.endswith('cvParam')]

                # handle cv blocks
                array_type = None
                dtype = None
                for cv in cvparams:
                    acc = cv.attrib.get('accession')
                    if acc == "MS:1000523":
                        dtype = np.float64
                    elif acc == "MS:1000521": 
                        dtype = np.float32
                    if acc == "MS:1000515":
                        array_type = "intensity_array"
                        int_count += 1
                    elif acc == "MS:1000595":
                        array_type = "time_array"
                        time_count += 1
                    elif acc == "MS:1000786":
                        array_type = "nonstandard"
                
                # grab encoded data
                if array_type == "time_array" or array_type == "intensity_array":
                    encoded = bda.find('binary', namespaces).text
                    decoded = MzMLProcessor.decode_binary_data(encoded,dtype)

                    # generate time_map (col_idx: time)
                    if array_type == "time_array" and idx == 1:
                        for i,time in enumerate(decoded):
                            time_map[i] = float(time)

                    # add intensity data to array
                    elif array_type == "intensity_array":
                        if idx != 0:
                            matrix[idx-1] = decoded
                        # add TIC to the end of the matrix
                        else:
                            matrix[-1] = decoded

        # convert ion map to sorted list
        mzs = [ion for ion,idx in sorted(ion_map.items(), key=lambda x: x[1])]

        # create intensity matrix object and return
        output_matrix = IntensityMatrix(intensity_matrix=matrix,unique_mzs=mzs,spectra_name=file_name,spectra_metadata=time_map)
        return output_matrix
    
    @staticmethod
    def create_intensity_matrix(mzml_path):
        """
        Generatews intensity matrix from mzml object, automatically detecting if it is SCAN or SIM
        Params:
            mzml_path                       Path to mzml object to analyze
        """

        # get aquisition type (SCAN or SIM)
        aq_type = MzMLProcessor.aq_type(mzml_path)

        if aq_type == "SIM":
            matrix = MzMLProcessor.create_sim_matrix(mzml_path)
        elif aq_type == "SCAN":
            matrix = MzMLProcessor.create_scan_matrix(mzml_path)

        return matrix

    @staticmethod
    def aq_type(mzml_path: Path):
        """
        Determines if the mzML file supplied is from a SIM or SCAN run
        Params:
            mzml_path                       Path to the mzML object to be analyzed
        """

        tree = ET.parse(mzml_path)
        root = tree.getroot()

        namespaces = {
            '': 'http://psi.hupo.org/ms/mzml'
        }

        # get filecontent information
        try:
            content = root.find('.//fileDescription/fileContent',namespaces)
        except Exception as e:
            raise ValueError(f"No file content found at {mzml_path}\nError:\n{e}")

        # get cvParams
        cvparams = [child for child in list(content) if child.tag.endswith('cvParam')]

        # iterate and save accession values
        for cv in cvparams:
            acc = cv.attrib.get("accession")
            if acc == "MS:1001472":
                return "SIM"
            elif acc == "MS:1000579":
                return "SCAN"
                

