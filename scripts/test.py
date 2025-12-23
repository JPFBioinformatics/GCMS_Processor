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

cfg = ConfigLoader(root_dir / "config.yaml")
file = "C:\Jack\code\GCMS_Automation\data\Int MET Scan 15.D"

MP = mp(cfg)