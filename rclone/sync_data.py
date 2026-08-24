

#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import subprocess
from plus_slurm import Job

#%%

class SyncMRI(Job):
		def run(self):

			rclone_cmd = ("rclone copy "
				 "mri:/data_MRI/neurokog/NCC25/analyze_fin "
				 "/mnt/ceph/groups_hdd/SCCGroup/salzburg_brain_dynamics/reabt/ncc/MRI/sync "
				#  "--filter '+ ????????????/NCC/firstLevel_sensory_M1C/**' "
				#  "--filter '+ ????????????/NCC/prepro_V1B/**' "
				 "--filter '+ ????????????/orig_functionals_trimmed/**' "
				 "--filter '+ ????????????/orig_structurals/**' "
				 "--filter '- *' -v")

			print('rclone command: \n', rclone_cmd, flush=True)

			result = subprocess.run(rclone_cmd, shell=True, capture_output=True, text=True)
			if result.returncode != 0:
				print(f"        Error occurred while copying data : \n        {result.stderr}")

class SyncMEG(Job):
		def run(self):

			rclone_cmd = ('rclone copy ' +
								 'sinuhe:/mnt/sinuhe/data_raw/aw_ncc/subject_subject ' +
								 '/home/scc_e_393956/Desktop/reabt/ncc/MEG/raw '+
								 '--exclude "/**/*log" '+
								 '--exclude "/**/*sss.fif" ' + 
								 '--transfers 10 ' +   
								 '-v')
			
			print(rclone_cmd, flush=True )

			result = subprocess.run(rclone_cmd, shell=True, capture_output=True, text=True)
			if result.returncode != 0:
				print(f"        Error occurred while copying data : \n        {result.stderr}")


class SyncEmptyRoom(Job):
		def run(self):

			rclone_cmd = ('rclone copy ' +
								 'sinuhe:/mnt/sinuhe/data_raw/empty_room/subject_subject ' +
								 '/home/scc_e_393956/Desktop/reabt/ncc/MEG/empty_room '+
								 '--filter "- /230413/**" ' +
								 '--filter "+ {23*,24*,25*,26*}/*.fif" ' +
								 ' --filter "- *"  ' +
								 '--transfers 10 ' +   
								 '-v')
			
			print(rclone_cmd, flush=True )

			result = subprocess.run(rclone_cmd, shell=True, capture_output=True, text=True)
			if result.returncode != 0:
				print(f"        Error occurred while copying data : \n        {result.stderr}")