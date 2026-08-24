import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

def fix_spm_rawdata_paths(spm: object, 
                    root_dir: str, 
                    subfolder_in_Subject = MRI_PREPROC_FOLDER,
                    corrected_prefix = 'swaudbFN_trimmed_') -> None:
    new_paths = []
    for p in spm.rawdata_files:
        subject = p.split('/')[-3]
        run_info_parts = p.split('/')[-1].split(',')
        base_name = run_info_parts[0]
        vol_index = run_info_parts[1] if len(run_info_parts) > 1 else '1'
        corrected_name = f'{corrected_prefix}{subject}_{base_name}'
        corrected_path = f'{root_dir}/{subject}/{subfolder_in_Subject}/{corrected_name},{vol_index}'
        new_paths.append(corrected_path)
    spm.rawdata_files = new_paths