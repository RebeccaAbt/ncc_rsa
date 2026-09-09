import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from glob import glob
import joblib
from contextlib import redirect_stdout
import warnings
import sys
from nibabel import load as nibload
import numpy as np
from utils.plots import plot_rdm

'''
Config used for the MRI data for modelbased fMRI/MEG fusion for the NCC study. 
Creating a class instance using a wildcard as subjects ID ("subjectID = '*'") 
is useful to obtain filename patterns instead of subject-specific file names.
The same is achieved when using the "load_config_instance" function from my
"utils.load_cfg" module without a subjectID input. This will automatically use 
the '*' as subjectID.
'''

class MRIconfig_Base:
	def __init__(self, subjectID='not_defined', maskNr=0, modelType = 'both'):
		self.subjectID = subjectID
		self.maskNr = maskNr
		self.modelType = modelType

		self.prefix = ''
		self.nCond = 24
		self.firstLevelModel = 'SM1C'
		self.firstLevelDir = MRI_1ST_LEVEL_FOLDER # before, this was 'firstLevel_sensory_M1B', but the data was computed with the wrong Outro Timestamp duration
		self.modelType = f'all{len(ALL_MODELS)}' # indicate total number of models
		self.RDMmethod = None
		self.RSAmethod = None
		self.SLradius = 2
		self.SLthr = 1
		self.resultsPlot_thr = 90
		self.replace_missing = 'imputation'

		self.dataDir =  MRI_DATA_DIR
		self.modelsDir = f'{CODE_DIR}/resources/'
		self.rawDataDir = MRI_RAW_DIR
		self.rsaFolder = 'rsa'
		# self.rsaFolder = 'rsa/M1B_wrongOutroTime' # to access the results computed with the wrong Outro Screen timing

		self.configure_paths()


	def configure_paths(self):

		# self.spmDir = f'{MRI_RAW_DIR}/{self.subjectID}/NCC/{self.firstLevelDir}/' #<--- We are using this path to directly access the data on the server without having to save it locally first. Only works, when the MRI server is mounted.
		self.spmDir = os.path.join(self.rawDataDir, self.subjectID, self.firstLevelDir)
		self.masksDir = os.path.join(MRI_MASKS_DIR, self.subjectID)
		# self.masksDir = get_spm_dir(self.subjectID)
		self.maskMargin = self.SLradius

		self.prefix_partial = f'{self.prefix}_partial'
		self.prefix_full = f'{self.prefix}_full'
		self.thr_string = str(self.SLthr).replace('.', '_')

		self.models_6_file = os.path.join(self.modelsDir, 'models_6.joblib')
		self.models_24_file = os.path.join(self.modelsDir, 'models_24.joblib')

		self.filePrefix = f'{self.prefix}_{self.firstLevelModel}_{self.nCond}Cond_{self.RDMmethod}_{self.RSAmethod}_r{self.SLradius}_thr{self.thr_string}_{self.modelType}'
		self.filePrefix_noModel = f'{self.prefix}_{self.firstLevelModel}_{self.nCond}Cond_{self.RDMmethod}_{self.RSAmethod}_r{self.SLradius}_thr{self.thr_string}'
		# self.filePrefix_partialMasks = f'{self.prefix}_partial_{self.maskNr}_{self.firstLevelModel}_{self.nCond}Cond_{self.RDMmethod}_{self.RSAmethod}_r{self.SLradius}_thr{self.thr_string}_{self.modelType}'
		self.filePrefix_partialMasks 				= f'{self.prefix}_partial_{self.maskNr}_{self.firstLevelModel}_{self.nCond}Cond_{self.RDMmethod}_r{self.SLradius}_thr{self.thr_string}_{self.modelType}'
		self.filePrefix_partialMasks_noModel 		= f'{self.prefix}_partial_{self.maskNr}_{self.firstLevelModel}_{self.nCond}Cond_{self.RDMmethod}_r{self.SLradius}_thr{self.thr_string}'
		self.filePrefix_partialMasks_noModel_rsa 	= f'{self.prefix}_partial_{self.maskNr}_{self.firstLevelModel}_{self.nCond}Cond_{self.RDMmethod}_{self.RSAmethod}_r{self.SLradius}_thr{self.thr_string}'
		self.filePrefix_fullBrain 				= f'fullBrain_{self.prefix}_{self.firstLevelModel}_{self.nCond}Cond_{self.RDMmethod}_{self.RSAmethod}_r{self.SLradius}_thr{self.thr_string}_{self.modelType}'
		self.filePrefix_fullBrain_noModel 		= f'fullBrain_{self.prefix}_{self.firstLevelModel}_{self.nCond}Cond_{self.RDMmethod}_r{self.SLradius}_thr{self.thr_string}'
		self.filePrefix_fullBrain_noModel_rsa 	= f'fullBrain_{self.prefix}_{self.firstLevelModel}_{self.nCond}Cond_{self.RDMmethod}_{self.RSAmethod}_r{self.SLradius}_thr{self.thr_string}'
   
		self.outDir = os.path.join(self.dataDir, f'{self.rsaFolder}/{self.__class__.__name__}/{self.subjectID}/')
		self.outDir_inference = os.path.join(self.dataDir, f'{self.rsaFolder}/{self.__class__.__name__}/') # for group-level data
	  
		# support modelType as string (e.g. 'all3'), a single model name, or a list of model names
		if isinstance(self.modelType, list):
			# list of model names -> store list of indices
			self.modelIdx = [ALL_MODELS.index(mt) for mt in self.modelType]
		elif isinstance(self.modelType, str) and 'all' in self.modelType:
			self.modelIdx = None
		else:
			self.modelIdx = ALL_MODELS.index(self.modelType) # --------------------------- This is the new indexing method, in case I want to add more models 

		# Moved from get_paths to here
		self.workspace_outFile = os.path.join(self.outDir, f'{self.filePrefix_partialMasks_noModel}_workspace.pkl')

		if self.maskNr == 0:
			self.DistPlotFile = os.path.join(self.outDir, f'{self.filePrefix_fullBrain}_distPlot.png')
			self.ResultsPlotFile = os.path.join(self.outDir, f'{self.filePrefix_fullBrain}_resultsPlot.png')
		else:
			self.DistPlotFile = os.path.join(self.outDir, f'{self.filePrefix_partialMasks}_distPlot.png')
			self.ResultsPlotFile = os.path.join(self.outDir, f'{self.filePrefix_partialMasks}_resultsPlot.png')


		self.plot1_title = f'Distribution | model: {self.firstLevelModel} | RDM: {self.RDMmethod} | RSA: {self.RSAmethod} - {self.subjectID}'
		self.plot2_title = f'Results | model: {self.firstLevelModel} | RDM: {self.RDMmethod} | RSA: {self.RSAmethod} - {self.subjectID}'

	def get_model_RDM(self):
		models = joblib.load(self.models_6_file if self.nCond == 6 else self.models_24_file)
		if self.modelType == 'sensory':
			if self.nCond == 6:
				raise ValueError("Sensory model not valid with 6 conditions.")
		if 'all' in self.modelType:
			print('returning all models', flush=True)
			return models
		else:
			if not isinstance(self.modelType, list):
				self.modelType = [self.modelType]
			print(f'returning the models for modeltype {models}', flush=True)
			return [models[ALL_MODELS.index(model)] for model in self.modelType]
		# single model name
		# 	return models[0]
		# elif self.modelType == 'suprasensory':
		# 	return models[1]
		# elif self.modelType == 'both':
		# 	return models
		# else:
		# 	raise ValueError(f"Unknown model type: {self.modelType}")

		
	def plot_model_RDM(self):
		models = joblib.load(self.models_6_file if self.nCond == 6 else self.models_24_file)
		if self.modelType == 'sensory':
			if self.nCond == 6:
				raise ValueError("Sensory model not valid with 6 conditions.")
			model = models[0]
		elif self.modelType == 'suprasensory':
			model = models[1]
		else:
			raise ValueError(f"Unknown model type: {self.modelType}")
		
		plot_rdm(model)
		

	def get_outFile_names(self):
		self.configure_paths()

		return {
			'info': os.path.join(self.outDir, f'{self.filePrefix_noModel}_info.pkl'),
			'centers_partial': os.path.join(self.outDir, f'{self.filePrefix_partialMasks_noModel}_centers.pkl'), 
			'neighbors_partial': os.path.join(self.outDir, f'{self.filePrefix_partialMasks_noModel}_neighbors.pkl'),
			'SL_rdms_partial': os.path.join(self.outDir, f'{self.filePrefix_partialMasks_noModel}_SL_rdms.pkl'), # 'both' will be displayed as model --> When Analysis pipeline is complete: change to not include model, because model not relevant for SL_RDMs!!
			'RDM_brain_partial': os.path.join(self.outDir, f'{self.filePrefix_partialMasks}_RDM_brain.pkl'),
			'eval_results_partial': os.path.join(self.outDir, f'{self.filePrefix_partialMasks}_eval_results.pkl'), # no model required, since it is direct output from SL analysis and automatically includes all models
			'eval_score_partial': os.path.join(self.outDir, f'{self.filePrefix_partialMasks}_eval_score.pkl'),

			'centers': os.path.join(self.outDir, f'{self.filePrefix_fullBrain_noModel}_centers.pkl'),
			'neighbors': os.path.join(self.outDir, f'{self.filePrefix_fullBrain_noModel}_neighbors.pkl'),
			'SL_rdms': os.path.join(self.outDir, f'{self.filePrefix_fullBrain_noModel}_SL_rdms.pkl'), # 'both' will be displayed as model
			'RDM_brain': os.path.join(self.outDir, f'{self.filePrefix_fullBrain}_RDM_brain.pkl'),
			'eval_results': os.path.join(self.outDir, f'{self.filePrefix_fullBrain}_eval_results.pkl'), # model-specific, since we compiled from partial masks separately for each model
			'eval_score': os.path.join(self.outDir, f'{self.filePrefix_fullBrain}_eval_score.pkl'),
			}
	
	def get_mask_file(self):
		
		if self.maskNr == 0: # if no partial brain mask was defined
			if self.subjectID == '*':
				print('\nDoing the "get_mask_file" method (probably while creating an instance of the MRI config class.No subjectID specified. Returning mask of subject "19910823ssld" as default')
				return os.path.join(self.rawDataDir, EXAMPLE_SUBJ_1, self.firstLevelDir, 'mask.nii')
				# return os.path.join(self.dataDir, 'masks', '19910823ssld', 'mask.nii')
			else:
				return os.path.join(self.spmDir, 'mask.nii') # using the full brain mask # don't use this when spmDir is a mounted folder, because mounting often doesn't work properly and we need the masks files often
				# return f'/home/reabt/experiments/ncc/MRI/data/masks_{self.firstLevelDir}/{self.subjectID}_mask.nii' 
				# return f'{self.masksDir}/mask.nii'
		
		else: # get partial brain mask
			# 'suprasensory' first-level data --> 4 stimuli per modality together as one condition
			if self.nCond == 6: 
				if self.maskMargin == 0:
					return os.path.join(self.masksDir, f'mask_part_{self.maskNr}.nii')
				
				elif isinstance(self.maskMargin, int) and 1 <= self.maskMargin <= 7:
					return os.path.join(self.masksDir, f'SL_marg{self.maskMargin}_mask_part_{self.maskNr}.nii')
				
				else:
					raise ValueError(f"Unknown mask type: {self.maskMargin}. Please set margin to an integer between 0 and 7")
				
			# 'sensory' first-level data --> 4 stimuli per modality as separate conditions
			elif self.nCond == 24:
				if self.maskMargin == 0:
					return os.path.join(self.masksDir, f'24cond_mask_part_{self.maskNr}.nii')
				
				elif isinstance(self.maskMargin, int) and 1 <= self.maskMargin <= 7:
					return os.path.join(self.masksDir, f'24cond_SL_marg{self.maskMargin}_mask_part_{self.maskNr}.nii')
				
				else:
					raise ValueError(f"Unknown mask type: {self.maskMargin}. Please set margin to an integer between 0 and 7")
				
	def get_centers_mask(self):
		centers = joblib.load(self.get_outFile_names()['SL_rdms']).rdm_descriptors['voxel_index']
		mask =nibload(self.get_mask_file()).get_fdata()

		centers_mask = np.zeros(mask.shape, dtype=int)
		coords = np.unravel_index(centers, mask.shape)
		centers_mask[coords] = 1
		return centers_mask.astype(bool)

	def print_summary(self):
		summary=(
		"\n------Configuration Summary ------\n"
		f"We are using the First Level Model: > {self.firstLevelModel} < from directory {self.firstLevelDir}\n"
		f"{self.nCond} conditions were defined\n"
		f"Model RDM Type:     {self.modelType}\n"
		f"RDM Method:         {self.RDMmethod}\n"
		f"RSA Method:         {self.RSAmethod}\n"
		f"Searchlights have a radius of > {self.SLradius} < voxels, with a threshold of > {self.SLthr} <\n"
		)
		if self.maskNr == 0:
			summary = (summary+f'Mask used: > full brain mask <\n')
		else: 
			summary = (summary+
					   f"Partial brain mask Nr.  {self.maskNr} used with a margin of {self.maskMargin} voxels\n"
					   "\n----------------------------------\n")
		print(summary)

	def save_summary(self):
		os.makedirs(os.path.join(self.dataDir, self.rsaFolder), exist_ok=True)
		filepath = os.path.join(self.dataDir, self.rsaFolder, f'{self.__class__.__name__}.txt')
		with open(filepath, 'w') as f:
			with redirect_stdout(f):
				self.print_summary()


# ------------------------------------------------------------------------ sensory-level conditions --> 24 conditions
# -------------------------------------------------------------------- euclidean
class MRIconfig_E2(MRIconfig_Base):
	def __init__(self, subjectID='not_defined', maskNr=0):
		super().__init__(subjectID, maskNr)
		self.prefix = 'SL_E2'
		self.nCond = 24
		# self.firstLevelModel = 'SM1B'
		# self.firstLevelDir = 'firstLevel_sensory_M1B'
		self.firstLevelModel = 'SM1C'
		self.firstLevelDir = MRI_1ST_LEVEL_FOLDER # before, this was 'firstLevel_sensory_M1B', but the data was computed with the wrong Outro Timestamp duration
		self.modelType = f'all{len(ALL_MODELS)}' # indicate total number of models
		self.RDMmethod = 'euclidean'
		self.RSAmethod = 'spearman'
		self.SLradius = 2
		self.SLthr = 0.5
		self.configure_paths()

class MRIconfig_E5(MRIconfig_Base):
	def __init__(self, subjectID='not_defined', maskNr=0):
		super().__init__(subjectID, maskNr)
		self.prefix = 'SL_E5'
		self.nCond = 24
		# self.firstLevelModel = 'SM1B'
		# self.firstLevelDir = 'firstLevel_sensory_M1B'
		self.firstLevelModel = 'SM1C'
		self.firstLevelDir = MRI_1ST_LEVEL_FOLDER # before, this was 'firstLevel_sensory_M1B', but the data was computed with the wrong Outro Timestamp duration
		self.modelType = f'all{len(ALL_MODELS)}' # indicate total number of models
		self.RDMmethod = 'euclidean'
		self.RSAmethod = 'spearman'
		self.SLradius = 5
		self.SLthr = 0.5
		self.configure_paths()

# -------------------------------------------------------------------- crossnobis

class MRIconfig_C2(MRIconfig_Base):
	def __init__(self, subjectID='not_defined', maskNr=0):
		super().__init__(subjectID, maskNr)
		self.prefix = 'SL_C2'
		self.nCond = 24
		self.SLradius = 2 # same as base
		self.SLthr = 1
		# self.firstLevelModel = 'SM1B'
		# self.firstLevelDir = 'firstLevel_sensory_M1B'
		self.firstLevelModel = 'SM1C'
		self.firstLevelDir = MRI_1ST_LEVEL_FOLDER # before, this was 'firstLevel_sensory_M1B', but the data was computed with the wrong Outro Timestamp duration
		self.modelType = f'all{len(ALL_MODELS)}' # indicate total number of models
		self.RDMmethod = 'crossnobis'
		self.RSAmethod = 'cosine_cov'
		self.configure_paths()


class MRIconfig_C5(MRIconfig_Base):
	def __init__(self, subjectID='not_defined', maskNr=0):
		super().__init__(subjectID, maskNr)
		self.prefix = 'SL_C5'
		self.nCond = 24
		self.SLradius = 5 # same as base
		self.SLthr = 1
		# self.firstLevelModel = 'SM1B'
		# self.firstLevelDir = 'firstLevel_sensory_M1B'
		self.firstLevelModel = 'SM1C'
		self.firstLevelDir = MRI_1ST_LEVEL_FOLDER # before, this was 'firstLevel_sensory_M1B', but the data was computed with the wrong Outro Timestamp duration
		self.modelType = f'all{len(ALL_MODELS)}' # indicate total number of models
		self.RDMmethod = 'crossnobis'
		self.RSAmethod = 'cosine_cov'
		self.configure_paths()


class MRIconfig_C5full(MRIconfig_C5):
	def __init__(self, subjectID='not_defined', maskNr=0):
		super().__init__(subjectID, maskNr)
		self.SLthr = 0.5

class MRIconfig_C2full(MRIconfig_C2):
	def __init__(self, subjectID='not_defined', maskNr=0):
		super().__init__(subjectID, maskNr)
		self.SLthr = 0.5


class MRIconfig_C2_nan(MRIconfig_C2):
	def __init__(self, subjectID='not_defined', maskNr=0):
		super().__init__(subjectID, maskNr)
		self.SLthr = 1
		self.replace_missing = 'nan'

class MRIconfig_C5_nan(MRIconfig_C5):
	def __init__(self, subjectID='not_defined', maskNr=0):
		super().__init__(subjectID, maskNr)
		self.SLthr = 1
		self.replace_missing = 'nan'


class MRIconfig_C2full_nan(MRIconfig_C2):
	def __init__(self, subjectID='not_defined', maskNr=0):
		super().__init__(subjectID, maskNr)
		self.SLthr = 0.5
		self.replace_missing = 'nan'

class MRIconfig_C5full_nan(MRIconfig_C5):
	def __init__(self, subjectID='not_defined', maskNr=0):
		super().__init__(subjectID, maskNr)
		self.SLthr = 0.5
		self.replace_missing = 'nan'


#%% tmp
class MRIconfig_C5test(MRIconfig_C5):
	def __init__(self, subjectID='not_defined', maskNr=0):
		super().__init__(subjectID, maskNr)

class MRIconfig_C2test(MRIconfig_C2):
	def __init__(self, subjectID='not_defined', maskNr=0):
		super().__init__(subjectID, maskNr)