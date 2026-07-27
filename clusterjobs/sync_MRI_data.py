

#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

import subprocess
from plus_slurm import Job

#%%

class DataSync(Job):
		def run(self):

			rclone_cmd = "rclone copy   mri:/data_MRI/neurokog/NCC25/analyze_fin   /mnt/ceph/groups_hdd/SCCGroup/salzburg_brain_dynamics/reabt/ncc/MRI/ --filter '+ ????????????/NCC/firstLevel_sensory_M1C/**' --filter '+ ????????????/NCC/prepro_V1B/**' --filter '- *' -v"

			result = subprocess.run(rclone_cmd, shell=True, capture_output=True, text=True)
			if result.returncode != 0:
				print(f"        Error occurred while copying data : \n        {result.stderr}")