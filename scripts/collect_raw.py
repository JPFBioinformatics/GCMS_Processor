# region Imports
import sys
from pathlib import Path

# location of pipeline root dir
root_dir = Path(__file__).resolve().parent.parent
# tell python to look here for modules
sys.path.insert(0, str(root_dir))

from src.config_loader import ConfigLoader
from src.intensity_matrix import IntensityMatrix as im
from src.mzml_processor import MzMLProcessor as mp
from src.report_generator import ReportGenerator as rg

# endregion

"""
Script to collect raw data and save it to an excel file
"""

def main():
    
    # load configs
    cfg = ConfigLoader(root_dir / "config.yaml")
    molecules,mzs,rts = cfg.load_collection_info()

    # convert .d files to mzML and pull intensity matrix objects
    processor = mp(cfg)
    matrices = processor.full_bulk_convert()

    # collect data
    output = im.collect_data(matrices,molecules,mzs,rts)
    
    # create report generator object
    report = rg(cfg,output)
    # generate data matrix
    report.generate_matrix(molecules)
    # write to excel file
    report.write_to_excel("raw")
    print("Excel report written")

if __name__ ==  "__main__":
    main()