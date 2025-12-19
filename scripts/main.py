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


def main():
    """
    Script that will convert .d agilant dirs to mzml files, parse them and remove an intensity matrix for each file, identify peaks of that intensity matrix,
    collect area of the peaks that are 
    """
    # load configs
    cfg = ConfigLoader(root_dir / "config.yaml")
    molecules,mzs,rts,case,control = cfg.load_collection_info()
    
    # convert .d files to mzML
    processor = mp(cfg)
    _,matrices = processor.full_bulk_convert()

    # collect data
    output = im.collect_data(matrices,molecules,mzs,rts)
    
    # generate report
    report = rg(output)

    # if case/control are sepcified then assign groups
    if case and control:
        report.assign_group(case,control)

    # generate a matrix of data for pca plotting
    report.generate_matrix()

    # generate qc plots
    report.qc_plots()

    # generate PCA plots
    report.calculate_pca()

    # write to excel file
    report.write_to_excel()

    # generate report
    report.generate_report()
    

if __name__ ==  "__main__":
    main()