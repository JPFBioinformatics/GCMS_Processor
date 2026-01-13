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

def main():
    
    @staticmethod
    def create_sim_matrix(mzml_path):

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
        mzs = [ion for ion,idx in sorted(ion_map.items(), key=lambda x: x[1])]

        # create intensity matrix object and return
        output_matrix = IntensityMatrix(intensity_matrix=matrix,unique_mzs=mzs,spectra_name=file_name,spectra_metadata=time_map)
        return output_matrix

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

    mzml_path = Path(r"C:\Jack\Projects\Danica SD 2025\11_25_25 Plasma\Sims\results\mzml_files\Danica SIM Sample 1.mzML")
    times,ions,matrix = create_sim_matrix(mzml_path)
    print(f"Ions (rows): {len(ions)}\nTime Points (cols): {len(times)}\nMatrix (rows,cols): {matrix.shape}")

if __name__ ==  "__main__":
    main()