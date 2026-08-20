"""

Handles converting agilant .D directories to mzml files and then converting those into IntensityMatrix
objects.

"""

# region Imports

import subprocess, base64, zlib, os, psutil, random
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from src.intensity_matrix import IntensityMatrix
from src.config_loader import ConfigLoader
from src.utils import log_subprocess,delete_file
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils import get_run_dir, configure_run_logging

# logging
import logging
logger = logging.getLogger(__name__)

# endregion

def full_bulk_convert(input_dir: Path, file_type: str, cfg):
    """
    Converts all compatible .d files in a directory to .mzml files and saves them to a directory in the input directory
    Returns:
        mzml_dir                        location of mzml dir
        matrices                        list of intensitymatrix objects created from all files in mzml
    """
    # stop any orphaned msconvert calls
    kill_orphaned_msconvert()

    # get input dir
    input_dir = Path(input_dir)

    # setup loger
    run_dir = get_run_dir(cfg.get("project_name"), cfg.get("run_name"))
    configure_run_logging(run_dir)

    # check to see if mzml files are already converted
    if file_type == '.D':
        raw_files = list(input_dir.glob("*.D"))
        tmpdir = input_dir / 'mzML_files'
        for raw_file in raw_files:
            if raw_file.is_dir():
                cmd = [
                    "msconvert",
                    str(raw_file),
                    "--mzML",
                    "--outdir", str(tmpdir)
                ]
                proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                try:
                    proc.wait()
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait()
        files = list(tmpdir.glob("*.mzML"))

    elif file_type == '.mzML':
        files = list(input_dir.glob("*.mzML"))

    # sort mzml files
    print(f"mzML files processed")
    files = sorted(files, key=lambda f: f.stem)

    # determine max workers for this system
    max_workers, results, success_count, fail_count = choose_max_workers(files, cfg, calibration_n=3, headroom_gb=2)
    remaining_files = [f for f in files if f not in results]

    run_dir = get_run_dir(cfg.get("project_name"), cfg.get("run_name"))

    with ProcessPoolExecutor(max_workers=max_workers, initializer=configure_run_logging, initargs=(run_dir,)) as executor:
        futures = {executor.submit(create_intensity_matrix, file, cfg): file for file in remaining_files}
        for future in as_completed(futures):
            file = futures[future]
            try:
                results[file] = future.result()
                logger.info(f"Created {file.name} IntensityMatrix")
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to process {file.name}: {e}")
                fail_count += 1

    matrices = [results[file] for file in files if file in results]

    logger.info(
        "\n\n-------------------- Converstion to IntensityMatrix --------------------\n"
        f"Total Converstions Attempted: {len(files)}\nSuccessful Conversions: {success_count}\nFailed Conversions: {fail_count}\nMatrices Output {len(matrices)}\n\n"
    )

    return matrices

def decode_binary_data(encoded_data, dtype, max_signal=None):
    """
    Decodes base64, decompresses zlib, and converts to a NumPy array with an associated m/z and time lists.
    Params:
        encoded_data                base64 data to be decoded
        dtype                       type of data that is converted
        max_signal                  maximum signal expected, if signal exceeds this it is set to 0
    """
    try:
        decoded = base64.b64decode(encoded_data)
        decompressed = zlib.decompress(decoded)
    except Exception as e:
        print(f"Exception:\n{e}")
        return None
    
    # generate result array, removing nan and clipping to max signal
    result = np.frombuffer(decompressed,dtype=dtype)                        # generate result array
    mask = result > 1e9
    if mask.any():
        indices = np.where(mask)[0]
        logger.info(f"Large Raw Values:{result[indices]}\nIndex values:{indices}")
    
    result =np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)          # replace nan with 0
    if max_signal is not None:
        result[result > max_signal] = 0                                     # remove excessively high signals
    return result

def bin_masses(unique_mzs, intensity_matrix, max_mz, min_mz):
    """
    Bins masses -0.3 to +0.7 of integer values
    Params:
        unique_mzs                      list of m/z values to bin
        intensity_matrix                intensity matrix that corrosponds to unbinned m/z values for processing
        min/max mz                      mz range for this run, used to prevent corrupted data from entering analysis
    
    Returns:
        binned_mzs                      list of binned mz values
        binned_matrix                   intensity matrix that has been binned
    """

    if min_mz is None:
        min_mz = 50
    if max_mz is None:
        max_mz = 1000
    
    # change unique mz list to array
    mz_array = np.asarray(unique_mzs)

    # filter out invalid entries
    valid = (mz_array >= min_mz) & ~np.isnan(mz_array) & (mz_array <= max_mz)
    mz_array = mz_array[valid]
    intensity_matrix = intensity_matrix[valid,:]

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

def create_scan_matrix(mzml_path, cfg, apply_threshold = True):
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

    time_map = {}
    intensity_list = []
    unique_mzs = set()
    skipped = 0

    max_mz = cfg.get('max_mz')
    min_mz = cfg.get('min_mz')
    max_signal = cfg.get('max_signal')

    # get name of sample
    name = mzml_path.stem
    logger.info(f"Sample: {name} Identified")

    # Iterate over each <spectrum> element and get scan information
    for spectrum in root.findall('.//spectrum', namespaces):
        scan_id = spectrum.get('id')
        if scan_id:
            scan_id = scan_id.split('=')[-1]

        scan_start_time = None
        scan_list = spectrum.find('scanList', namespaces)
        if scan_list is not None:
            scan = scan_list.find('scan', namespaces)
            if scan is not None:
                cv = scan.find('.//cvParam[@name="scan start time"]', namespaces)
                if cv is not None:
                    scan_start_time = float(cv.get('value'))

        # find binary data arrays
        mz_encoded = None
        intensity_encoded = None

        for bda in spectrum.findall('./binaryDataArrayList/binaryDataArray', namespaces):
            array_type = None

            for cv in bda.findall('cvParam',namespaces):
                acc = cv.get('accession')
                if acc == "MS:1000514":
                    array_type = "mz"
                elif acc == "MS:1000515":
                    array_type = "intensity"

            binary = bda.find('binary',namespaces)
            if binary is None or not binary.text:
                continue
        
            # save encoded binary data
            if array_type == 'mz':
                mz_encoded = binary.text
            elif array_type == 'intensity':
                intensity_encoded = binary.text

        # if bianry data is not present, then skip this scan
        if mz_encoded is None or intensity_encoded is None:
            skipped += 1
            continue

        # decode the data
        mz_array = decode_binary_data(mz_encoded,dtype=np.float64, max_signal=max_signal)
        intensity_array = decode_binary_data(intensity_encoded,dtype=np.float32, max_signal=max_signal)

        # make sure array values are valid
        if mz_array is None or intensity_array is None:
            skipped += 1
            continue

        # Save metadata for the spectrum in the list
        time_map[len(intensity_list)] = float(scan_start_time)
        
        # ensure consistent lengths before zipping
        if len(mz_array) != len(intensity_array):
            skipped += 1
            continue

        # Create a dictionary for each spectrum (m/z -> intensity)
        spectrum_intensity_dict = dict(zip(mz_array, intensity_array))

        # Add the spectrum dictionary to the intensity_list
        intensity_list.append(spectrum_intensity_dict)

        # Add the m/z values to the set of unique m/z values
        unique_mzs.update(mz_array)

    logger.info(f"Sample: {name} Finsihed mzML parse")

    # show how many spectra have beens skipped
    if skipped > 0:
        logger.warning(f"File: {mzml_path.stem}\nSkipped: {skipped}\n")

    # sort unnizue mz lit
    unique_mzs = sorted(unique_mzs)

    # check max/min
    if min_mz is None:
        min_mz = 50
    if max_mz is None:
        max_mz = 1000

    # remove invalid mz values from list
    mz_array  = np.asarray(unique_mzs)
    valid = (mz_array >= min_mz) & ~np.isnan(mz_array) & (mz_array <= max_mz)
    mz_array = mz_array[valid]

    # assign bins
    bin_assignments = (mz_array + 0.3).astype(int)
    binned_mzs_arr, inverse = np.unique(bin_assignments, return_inverse=True)
    mz_to_bin_row = dict(zip(mz_array, inverse))

    # generate binnd matrix
    binned_matrix = np.zeros((len(binned_mzs_arr), len(intensity_list)))

    # fill in matrix from bins
    for col_idx, spectrum_intensity_dict in enumerate(intensity_list):
        for mz,intensity in spectrum_intensity_dict.items():
            bin_row = mz_to_bin_row.get(mz)
            if bin_row is not None:
                binned_matrix[bin_row, col_idx] += intensity
    binned_mzs = list(binned_mzs_arr)

    # add TIC row to end of matrix
    sum_row = np.sum(binned_matrix, axis=0)
    final_matrix = np.vstack((binned_matrix,sum_row))

    # add 9999 value to end of binned_mzs to represent the TIC
    binned_mzs.append(9999)

    # create intensity matrix object
    output_matrix = IntensityMatrix(intensity_matrix=final_matrix,
                                    unique_mzs=binned_mzs,
                                    cfg=cfg,
                                    sample_name=name,
                                    time_map=time_map,
                                    matrix_type="SCAN",
                                    detect_peaks=True,
                                    apply_threshold=apply_threshold)
    logger.info(f"Sample: {name} IntensityMatrix Created")

    time_vals = output_matrix.get_time_per_scan()
    logger.info(f"\nTotal Scans: {len(time_vals['array'])}\nAvg Time per Scan: {time_vals['avg']}\nStdev: {time_vals['stdev']}\nPct Err: {(100*time_vals['stdev']/time_vals['avg']):.2f}")

    return output_matrix

def create_sim_matrix(mzml_path, cfg):
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
                decoded = decode_binary_data(encoded,dtype)

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
    mzs = [ion for ion,_ in sorted(ion_map.items(), key=lambda x: x[1])]

    # get row varainces for time matrix and see if it looks good

    # create intensity matrix object and return
    output_matrix = IntensityMatrix(intensity_matrix=matrix,
                                    unique_mzs=mzs,
                                    cfg=cfg,
                                    sample_name=file_name,
                                    time_map=time_map,
                                    matrix_type="SIM",
                                    detect_peaks=True)
    logger.info(f"Produced inntensity matrix for sample: {file_name}")
    return output_matrix

def create_intensity_matrix(mzml_path, cfg):
    """
    Generatews intensity matrix from mzml object, automatically detecting if it is SCAN or SIM
    Params:
        mzml_path                       Path to mzml object to analyze
    """

    # get aquisition type (SCAN or SIM)
    type = aq_type(mzml_path)

    if type == "SIM":
        matrix = create_sim_matrix(mzml_path, cfg)
    elif type == "SCAN":
        matrix = create_scan_matrix(mzml_path, cfg)
    """
    peak_mem_gb = psutil.Process(os.getpid()).memory_info().peak_wset / 1e9
    logger.info(f"{mzml_path.stem}: peak working set so far {peak_mem_gb:.2f} GB")
    """

    return matrix

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

def kill_orphaned_msconvert():
    """
    kills leftover msconvert processed left over from a failed run
    """

    target_names = {
        "msconvert.exe"
    }

    killed_pids = []
    for proc in psutil.process_iter(['pid','name']):
        try:
            if proc.info['name'] and proc.info['name'].lower() in target_names:
                proc.terminate()
                killed_pids.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if killed_pids:
        procs = [psutil.Process(pid) for pid in killed_pids if psutil.pid_exists(pid)]
        _, alive = psutil.wait_procs(procs,timeout=5)
        for p in alive:
            p.kill()
        logger.warning(f"Cleaned up {len(killed_pids)} orphaned process(es) from previous run: {killed_pids}")

def create_im_with_mem(mzml_path, cfg):
    """
    creates an intesnitymatrix object and returns the peak memroy needed for process
    """
    matrix = create_intensity_matrix(mzml_path, cfg)
    peak_mem = psutil.Process(os.getpid()).memory_info().peak_wset
    return matrix,peak_mem

def choose_max_workers(files, cfg, calibration_n=3, headroom_gb=2.0):
    """
    chooses a random number of files to use to test system and calibrate how many workers
    we can use to process samples based on availbe cpu cores and memory
    """
    cpu_ceiling = max(1, os.cpu_count()-1)

    calibration_files = random.sample(files, min(calibration_n,len(files)))
    
    peak_mem_bytes = 0
    calibration_results  = {}
    success_count = 0
    fail_count = 0

    run_dir = get_run_dir(cfg.get("project_name"), cfg.get("run_name"))

    with ProcessPoolExecutor(max_workers=1, initializer=configure_run_logging, initargs=(run_dir,)) as executor:
        for file in calibration_files:
            try:
                matrix,mem = executor.submit(create_im_with_mem, file, cfg).result()
                peak_mem_bytes = max(peak_mem_bytes, mem)
                calibration_results[file] = matrix
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to process {file.name} during calibration: {e}")
                fail_count += 1
    
    if peak_mem_bytes == 0:
        logger.warning(f"Calbiration failed for all sampled files, defaulting to max_workers = 1")
        return 1, calibration_results, success_count, fail_count
        
    available_bytes = psutil.virtual_memory().available - headroom_gb * 1e9
    mem_ceiling = max(1, int(available_bytes // peak_mem_bytes))
    max_workers = min(cpu_ceiling, mem_ceiling)

    logger.info(
        "\n------------------------------ CPU Optimization ------------------------------\n"
        f"Total CPU Cores: {os.cpu_count()} | CPU Cores Available: {cpu_ceiling}\n"
        f"Memory Available (GB): {available_bytes/1e9:.2f} | Max Workers: {min(cpu_ceiling,mem_ceiling)}\n"
    )

    return max_workers, calibration_results, success_count, fail_count
