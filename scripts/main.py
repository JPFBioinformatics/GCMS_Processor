# region Imports
import sys,datetime
from pathlib import Path

# location of pipeline root dir
root_dir = Path(__file__).resolve().parent.parent
# tell python to look here for modules
sys.path.insert(0, str(root_dir))

from src.config_loader import ConfigLoader
from src.intensity_matrix import IntensityMatrix as im
from src.mzml_processor import MzMLProcessor as mp
from src.report_generator import ReportGenerator as rg
from src.utils import log_timestamp,delete_file,delete_directory

# endregion

def main():
    # get starting ts
    start_ts = datetime.datetime.now()

    # load configs
    cfg = ConfigLoader(root_dir / "config.yaml")
    indir = Path(cfg.get("input_dir"))
    results = Path(cfg.get("results_dir"))
    log_dir = indir / results
    log_dir.mkdir(parents=True,exist_ok=True)
    molecules,mzs,rts = cfg.load_collection_info()

    # delete old log files if you're rerunning the analysis
    log_file = log_dir / "timestamp_log.jsonl"
    sub_log = log_dir / "subprocess_log.jsonl"
    files = [log_file,sub_log]
    for file in files:
        delete_file(file)

    # log start time
    log_timestamp(log_dir,"start",start_ts)
    print("Start TS Logged")
    
    # convert .d files to mzML and pull intensity matrix objects
    processor = mp(cfg)
    matrices = processor.full_bulk_convert()
    ts1 = datetime.datetime.now()
    log_timestamp(log_dir,"mzml_convert",ts1)
    print("mzML Files Processed")

    # collect raw data
    output = im.collect_data(matrices,molecules,mzs,rts)
    ts2 = datetime.datetime.now()
    log_timestamp(log_dir,"raw_data_collection",ts2)
    print("Raw Data Collected")

    # generate report object
    report = rg(cfg,output)
    # generate data matrix
    report.generate_matrix(molecules)
    ts3 = datetime.datetime.now()
    log_timestamp(log_dir,"matrix_generation",ts3)
    print("Matrix Generated")

    # genearte normalized matrix
    report.normalize_matrix()
    ts4 = datetime.datetime.now()
    log_timestamp(log_dir,"matrix_normalized",ts4)
    print("Matrix Normalized")

    # write to excel file
    report.write_to_excel()
    ts5 = datetime.datetime.now()
    log_timestamp(log_dir,"excel_written",ts5)
    print("Excel written")

    # write qc report
    report.generate_report()
    ts6 = datetime.datetime.now()
    log_timestamp(log_dir,"report_generated",ts6)
    print("Report Generated")

    # delete mzML files for 
    flag = cfg.get("keep_mzml")
    if flag == True:
        mzml = cfg.get("mzml_dir")
        mzml_dir = log_dir / mzml
        delete_directory(mzml_dir)

    # log end time
    end_ts = datetime.datetime.now()
    log_timestamp(log_dir,"end",end_ts)
    print("End TS Logged")

if __name__ ==  "__main__":
    main()