import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import importlib

# This was before splitten MEG/MRI into separate configs:
def load_config_instance(config_class_name, subjectID = '*', maskNr = 0):
    config_module = importlib.import_module("configs.config")
    config_class = getattr(config_module, config_class_name)
    return config_class(subjectID, maskNr)


def load_MEG_config_instance(config_class_name, subjectID = '*'):
    config_module = importlib.import_module("configs.config_MEG")
    config_class = getattr(config_module, config_class_name)
    return config_class(subjectID)


def load_MRI_config_instance(config_class_name, subjectID = '*', maskNr = 0):
    config_module = importlib.import_module("configs.config_MRI")
    config_class = getattr(config_module, config_class_name)
    return config_class(subjectID, maskNr)


def load_fusion_config_instance(config_class_name, subjectID = '*'):
    config_module = importlib.import_module("configs.config_fusion")
    config_class = getattr(config_module, config_class_name)
    return config_class(subjectID)