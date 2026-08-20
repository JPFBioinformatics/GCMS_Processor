
# region Imports
from pathlib import Path
from datetime import datetime
from src.mzml_processor import create_scan_matrix
from src.config_loader import ConfigLoader
import json

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

from src.intensity_matrix import IntensityMatrix as IM

# endregion

# region logging

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=Path(__file__).parent / "gen_json_data.log"
)
logger = logging.getLogger(__name__)

# endregion

file_path = Path(r"C:\Jack\Projects\IlyaAura Mouse Labelling\7_10_int1\mzML_files\Int 1.mzML")
json_path = Path(f'im_data.json')

cfg_path = Path(__file__).parent / 'config.yaml'
cfg = ConfigLoader(cfg_path)
peak_mode = cfg.get('peak_mode')

starttime = datetime.now()

im = create_scan_matrix(file_path, cfg=cfg, apply_thresholds = False)

endtime = datetime.now()

# get data for histograms
sn_ratios = []
cwt_scores = []
cwt_scales = []
ridge_spans = []
heights = []
scales = []
scores = []
widths = []
height_thresholds = im.height_thresholds
for _,peak_list in im.peak_dict.items():
    for peak in peak_list:
        sn_ratios.append(float(peak['sn_ratio']))
        cwt_scores.append(float(peak['cwt_score']))
        cwt_scales.append(float(peak['cwt_scale']))
        ridge_spans.append(float(peak['ridge_span']))
        heights.append(float(peak['height']))
        widths.append(float(peak['right_bound'] - peak['left_bound']))

data = {
    'sn_ratios': sn_ratios,
    'cwt_scores': cwt_scores,
    'cwt_scales': cwt_scales,
    'ridge_spans': ridge_spans,
    'heights': heights,
    'widths': widths,
    'height_thresholds': height_thresholds,
    'mzs': [int(mz) for mz in im.unique_mzs],
    'first_derivs': im.first_derivs.tolist(),
    'second_derivs': im.second_derivs.tolist(),
    'smoothed_signal': im.smoothed_signal.tolist(),
    'cwt_max_scores': im.cwt_scores.tolist(),
    'cwt_max_scales': im.cwt_scales.tolist()
}

with open(json_path, 'w') as f:
    json.dump(data, f)