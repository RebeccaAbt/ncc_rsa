#%%
import os
import sys
sys.path.append(os.path.dirname('/home/scc_e_393956/ncc/rsa/'))
from configs.config2 import * # directories + constants
import subprocess

from plus_slurm import Job


#%%
class CloneSinuhe(Job):
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
			print(f"Error occurred while copying data from sinuhe: \n        {result.stderr}")
		else:
			print('Done!' )
		
