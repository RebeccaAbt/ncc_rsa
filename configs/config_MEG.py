import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

from glob import glob
import joblib
from contextlib import redirect_stdout
from utils.plots import plot_rdm

'''
Config used for the MEG data for modelbased fMRI/MEG fusion for the NCC study. 
Creating a class instance using a wildcard as subjects ID ("subjectID = '*'") 
is useful to obtain filename patterns instead of subject-specific file names.
The same is achieved when using the "load_config_instance" function from my
"utils.load_cfg" module without a subjectID input. This will automatically use 
the '*' as subjectID.
'''

class MEGconfig_Base:
    def __init__(self, subjectID='not_defined', modelType = 'both'):
        self.subjectID = subjectID
        self.modelType = modelType

        self.prefix = '' # End prefix with "-" if set to a string
        self.nCond = 24
        self.modelType = f'all{len(ALL_MODELS)}' # indicate total number of models
        self.RDMmethod = 'crossnobis'
        self.RSAmethod = 'cosine_cov'
        self.resultsPlot_thr = 90

        self.l_freq = None
        self.h_freq = 40
        self.t_min = -1
        self.t_max = 1
        self.fs_new = 100
        self.channels = 'meg'

        self.modelsDir = MODELS_DIR
        self.rawDataFolder = 'sync'
        self.rsaFolder = 'rsa'
        # self.rsaFolder = 'rsa/M1B_wrongOutroTime' # to access the results computed with the wrong Outro Screen timing

        self.dataDir = MEG_DATA_DIR
        self.dataFolder = 'epochs_clean/manual_finish'
        # self.filePattern = '*_clean-epo.fif'
        # self.filePattern = '*_maxfilter_ica_1-99Hz__fs_1000__\[-1.5_1.5\]s_detrend_1_meg_clean-epo.fif'
        self.filePattern = '*_maxfilter_ica_1-99Hz__fs_1000*-1.5_1.5*s_detrend_1_meg_clean-epo.fif'

        self.cv_descriptor = None

        self.configure_paths()


    def configure_paths(self):
        self.prefix = self.__class__.__name__.partition('_')[2]
        
        self.models_6_file = os.path.join(self.modelsDir, 'models_6.joblib')
        self.models_24_file = os.path.join(self.modelsDir, 'models_24.joblib')
        self.MEG_searchpath = os.path.join(self.dataDir, self.dataFolder, self.subjectID, self.filePattern)
        # print(f"MEG seachpath: {self.MEG_searchpath}", flush=True)
        self.MEG_inFile = glob(self.MEG_searchpath)
        
        if self.cv_descriptor is None:
            self.cv_info = ''
        else:
            self.cv_info = f'cv_{self.cv_descriptor}_'

    def get_model_RDM(self):
        models = joblib.load(self.models_6_file if self.nCond == 6 else self.models_24_file)
        if self.modelType == 'sensory':
            if self.nCond == 6:
                raise ValueError("Sensory model not valid with 6 conditions.")
            return models[0]
        elif self.modelType == 'suprasensory':
            return models[1]
        elif self.modelType == 'both':
            return models
        else:
            raise ValueError(f"Unknown model type: {self.modelType}")

        
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
            'movie': os.path.join(self.dataDir, f'movie_RDMs/{self.prefix}/{self.prefix}_rdm_movie_{self.cv_info}{self.subjectID}.pkl'), #remove number of config calss name (e-g- "E" instead of "E2" because for the MEG RDMs, the Searchlight radius is not relevant)
            'movie_mean': os.path.join(self.dataDir, f'movie_RDMs/{self.prefix}/rdm_movie_{self.cv_info}subj_mean.pkl'),
            'movie_eval': os.path.join(self.dataDir, f'movie_eval/{self.prefix}/model_eval_results_{self.cv_info}{self.subjectID}.pkl')
            }
    
    def print_summary(self):
        print("\n------Configuration Summary ------\n")
        print(f"{self.nCond} conditions were defined")
        print(f"Model RDM Type:     {self.modelType}")
        print(f"RDM Method:         {self.RDMmethod}")
        print(f"RSA Method:         {self.RSAmethod}")
        print(f"MEG cv_descriptor:  {self.cv_descriptor}")
        print("\n----------------------------------\n")

    def save_summary(self):
        filepath = os.path.join(self.dataDir, self.rsaFolder, f'{self.__class__.__name__}.txt')
        with open(filepath, 'w') as f:
            with redirect_stdout(f):
                self.print_summary()


# -------------------------------------------------------------------- euclidean

class MEGconfig_E(MEGconfig_Base):
    def __init__(self, subjectID='not_defined'):
        super().__init__(subjectID)
        # self.prefix = '' # End prefix with "-" if set to a string
        self.RDMmethod = 'euclidean'
        self.RSAmethod = 'spearman'
        self.cv_descriptor = None
        self.configure_paths()

# class MEGconfig_Eb(MEGconfig_E): # new MEG preprocessing
#     def __init__(self, subjectID='not_defined'):
#         super().__init__(subjectID)
#         self.dataDir = '/home/reabt/experiments/ncc/MEG/data'
#         self.dataFolder = 'epochs_potato'
#         self.filePattern = '*_MF__ICA__filter_1-30__fs_100__-1.0-1.0s_potato4grad__estim_oas__thres_z3_meg-epo_clean.fif'
#         self.configure_paths()

     
# -------------------------------------------------------------------- crossnobis


class MEGconfig_C(MEGconfig_Base):
    def __init__(self, subjectID='not_defined'):
        super().__init__(subjectID)
        self.prefix = '' # End prefix with "-" if set to a string
        self.RDMmethod = 'crossnobis'
        self.RSAmethod = 'cosine_cov'
        self.cv_descriptor = 'run_idx'
        self.configure_paths()


# class MEGconfig_Cb(MEGconfig_C): # new MEG preprocessing
#     def __init__(self, subjectID='not_defined'):
#         super().__init__(subjectID)
#         self.dataDir = '/home/reabt/experiments/ncc/MEG/data'
#         self.dataFolder = 'epochs_potato'
#         self.filePattern = '*_MF__ICA__filter_1-30__fs_100__-1.0-1.0s_potato4grad__estim_oas__thres_z3_meg-epo_clean.fif'
#         self.configure_paths()


