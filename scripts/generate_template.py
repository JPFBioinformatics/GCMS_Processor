# region Imports

import sys, subprocess, os
from pathlib import Path

# location of pipeline root dir
root_dir = Path(__file__).resolve().parent.parent
# tell python to look here for modules
sys.path.insert(0, str(root_dir))

from src.report_generator import ReportGenerator
from src.config_loader import ConfigLoader

# endregion

"""
Script that will generate a template file for the set of samples you want to run
"""

cfg = ConfigLoader(root_dir / "config.yaml")
template_name = cfg.get("template_file")
template = Path(cfg.get("input_dir")) / template_name

RG = ReportGenerator(cfg)

RG.generate_template()

# open the file
if sys.platform.startswith("win"):                  # widows
    os.startfile(template)
elif sys.platform.startswith("darwin"):             # mac
    subprocess.run(["open",template])
else:                                               # linux
    subprocess.run(["xdg-open",template])