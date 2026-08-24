import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from glob import glob
import joblib
from contextlib import redirect_stdout
import numpy as np
from copy import deepcopy

from utils.load_cfg import *
from utils.plots import plot_rdm
from utils.subj import get_fusion_subjects

'''
Config used for the MEG data for modelbased fMRI/MEG fusion for the NCC study. 
Creating a class instance using a wildcard as subjects ID ("subjectID = '*'") 
is useful to obtain filename patterns instead of subject-specific file names.
The same is achieved when using the "load_config_instance" function from my
"utils.load_cfg" module without a subjectID input. This will automatically use 
the '*' as subjectID.
'''

class FusionConfig_Base:
    def __init__(self, subjectID='*'):

        self.MEG_config = ''
        self.MRI_config = ''
        self.subjectID = subjectID
        self.modelType = 'both'

        self.fusionDir = FUSION_DATA_DIR
        self.outDir = f'{self.fusionDir}/{self.__class__.__name__}'
        self.min_MEG_trials = 0
        
        # Do not load configs here; subclasses will set MEG_config and MRI_config, then load configs and configure paths.


    def configure_paths(self):

        os.makedirs(self.outDir, exist_ok=True)

        self.prefix = self.__class__.__name__.partition('_')[2]

        self.MEG_cfg = load_MEG_config_instance(self.MEG_config, self.subjectID)
        self.MRI_cfg = load_MRI_config_instance(self.MRI_config, self.subjectID)

        self.MEG_input = self.MEG_cfg.get_outFile_names()['movie']
        self.MRI_input = self.MRI_cfg.get_outFile_names()['SL_rdms']

        self.maskFile = self.get_mask_file()

    def configure_cv(self):
        if self.MEG_cfg.cv_descriptor is None:
                self.cv_info = ''
        else:
            self.cv_info = f'cv_{self.MEG_cfg.cv_descriptor}_'
       

    def get_mean_movie(self):
        '''
        Compute the mean of the RDM-movies of sujects, that are currently included 
        in the fusion analysis (=subjects where MEG + fMRI data exists).
        Outputs a variable of the class "RDMs"
        '''

        all_subj = get_fusion_subjects()
        cfg = load_MEG_config_instance(self.MEG_config, all_subj[0]) 

        all_RDMs = []
        for subj in all_subj:

            cfg.subjectID = subj

            movieFile = cfg.get_outFile_names()['movie']

            if os.path.exists(movieFile):
                rdm_movie = joblib.load(movieFile)
                all_RDMs.append(rdm_movie.get_vectors())

        mean_rdms = np.mean(np.array(all_RDMs), axis = 0)
        mean_movie = deepcopy(rdm_movie)
        mean_movie.dissimilarities = mean_rdms # put the data back in necessary RDMs structure
        return mean_movie


    def get_outFile_names(self):
        self.configure_paths()
        
        os.makedirs(os.path.join(self.outDir, 'commonalities'), exist_ok=True)
        os.makedirs(os.path.join(self.outDir, 'stats'), exist_ok=True)

        return {
            'fusion_mat': os.path.join(self.outDir, 'commonalities', f'{self.prefix}_{self.subjectID}_fusion.mat'),
            'fusion_pkl': os.path.join(self.outDir, 'commonalities', f'{self.prefix}_{self.subjectID}_fusion.pkl'),
            'fusion_mean_mat': os.path.join(self.outDir, 'commonalities', f'{self.prefix}_{self.subjectID}_fusion_mean.mat'),
            'fusion_mean_pkl': os.path.join(self.outDir, 'commonalities', f'{self.prefix}_{self.subjectID}_fusion_mean.pkl'),
            'cp':  os.path.join(self.outDir, 'stats', f'{self.prefix}_cp.pkl'),
            'cp_mean':  os.path.join(self.outDir, 'stats', f'{self.prefix}_cp_mean.pkl'), # with fusion data using MEG-subject-mean
            'good_clusters':  os.path.join(self.outDir, 'stats', f'{self.prefix}_good_clusters.pkl'),
            'good_clusters_mean':  os.path.join(self.outDir, 'stats', f'{self.prefix}_good_clusters_mean.pkl')  # with fusion data using MEG-subject-mean
            }
    
    def get_mask_file(self):
        return self.MRI_cfg.get_mask_file()
    
    def get_model_RDM(self):
        return self.MRI_cfg.get_model_RDM()
    
    def print_summary(self):
        summary = (
            f"this Fusion uses the MRI config {self.MRI_cfg.__class__.__name__} "
            f"and the MEG config {self.MEG_cfg.__class__.__name__}\n"
            "_______________________________________________________________________________________________________________\n\n"
            "------MRI config Summary ------\n"
            f"We are using the First Level Model: > {self.MRI_cfg.firstLevelModel} < from directory {self.MRI_cfg.firstLevelDir}\n"
            f"{self.MRI_cfg.nCond} conditions were defined\n"
            f"Model RDM Type:     {self.MRI_cfg.modelType}\n"
            f"RDM Method:         {self.MRI_cfg.RDMmethod}\n"
            f"RSA Method:         {self.MRI_cfg.RSAmethod}\n"
            f"Searchlights have a radius of > {self.MRI_cfg.SLradius} < voxels, with a threshold of > {self.MRI_cfg.SLthr} <\n\n"
            "------MEG config Summary ------\n"
            f"{self.MEG_cfg.nCond} conditions were defined\n"
            f"Model RDM Type:     {self.MEG_cfg.modelType}\n"
            f"RDM Method:         {self.MEG_cfg.RDMmethod}\n"
            f"RSA Method:         {self.MEG_cfg.RSAmethod}\n"
            f"MEG cv_descriptor:  {self.MEG_cfg.cv_descriptor}\n"
            "_______________________________________________________________________________________________________________\n"
        )
        print(summary)

    def save_summary(self):
        filepath = os.path.join(self.fusionDir, f'{self.__class__.__name__}.txt')
        with open(filepath, 'w') as f:
            with redirect_stdout(f):
                self.print_summary()

# ======================================================================================= original configs
# -------------------------------------------------------------------- euclidean

class FusionConfig_E2(FusionConfig_Base):
    def __init__(self, subjectID='*'):
        super().__init__(subjectID)
        self.MEG_config = 'MEGconfig_E'
        self.MRI_config = 'MRIconfig_E2'
        self.cv_descriptor = None      
        self.configure_paths() # this also initiates the MEG/MRI config instances
        self.configure_cv()
        # if self.MEG_cfg.cv_descriptor is None:
        #     self.cv_info = ''
        # else:
        #     self.cv_info = f'cv_{self.MEG_cfg.cv_descriptor}_'


class FusionConfig_E5(FusionConfig_Base):
    def __init__(self, subjectID='*'):
        super().__init__(subjectID)
        self.MEG_config = 'MEGconfig_E'
        self.MRI_config = 'MRIconfig_E5'
        self.cv_descriptor = None
        self.configure_paths() # this also initiates the MEG/MRI config instances   
        self.configure_cv()

# -------------------------------------------------------------------- crossnobis

class FusionConfig_C2(FusionConfig_Base):
    def __init__(self, subjectID='*'):
        super().__init__(subjectID)
        self.MEG_config = 'MEGconfig_C'
        self.MRI_config = 'MRIconfig_C2'
        self.cv_descriptor = None
        self.configure_paths() # this also initiates the MEG/MRI config instances  11
        self.configure_cv()

class FusionConfig_C5(FusionConfig_Base):
    def __init__(self, subjectID='*'):
        super().__init__(subjectID)
        self.MEG_config = 'MEGconfig_C'
        self.MRI_config = 'MRIconfig_C5'
        self.cv_descriptor = None
        self.configure_paths() 
        self.configure_cv()


# ======================================================================================= 10 Trl limit configs
class FusionConfig_E2_10trl(FusionConfig_E2):
    def __init__(self, subjectID='*'):
        super().__init__(subjectID)
        self.min_MEG_trials = 10
        self.configure_paths()
        self.configure_cv()

class FusionConfig_E5_10trl(FusionConfig_E5):
    def __init__(self, subjectID='*'):
        super().__init__(subjectID)
        self.min_MEG_trials = 10
        self.configure_paths()
        self.configure_cv()


class FusionConfig_C2_10trl(FusionConfig_C2):
    def __init__(self, subjectID='*'):
        super().__init__(subjectID)
        self.min_MEG_trials = 10
        self.configure_paths() 
        self.configure_cv()

class FusionConfig_C5_10trl(FusionConfig_C5):
    def __init__(self, subjectID='*'):
        super().__init__(subjectID)
        self.min_MEG_trials = 10
        self.configure_paths() 
        self.configure_cv()

# =======================================================================================  Fusion with potato MEG data

class FusionConfig_E2b(FusionConfig_Base):
    def __init__(self, subjectID='*'):
        super().__init__(subjectID)
        self.MEG_config = 'MEGconfig_Eb'
        self.MRI_config = 'MRIconfig_E2'
        self.cv_descriptor = None
        self.configure_paths() 
        self.configure_cv()

class FusionConfig_E5b(FusionConfig_Base):
    def __init__(self, subjectID='*'):
        super().__init__(subjectID)
        self.MEG_config = 'MEGconfig_Eb'
        self.MRI_config = 'MRIconfig_E5'
        self.cv_descriptor = None
        self.configure_paths()
        self.configure_cv()


class FusionConfig_C2b(FusionConfig_Base):
    def __init__(self, subjectID='*'):
        super().__init__(subjectID)
        self.MEG_config = 'MEGconfig_Cb'
        self.MRI_config = 'MRIconfig_C2'
        self.cv_descriptor = None
        self.configure_paths() 
        self.configure_cv()


class FusionConfig_C5b(FusionConfig_Base):
    def __init__(self, subjectID='*'):
        super().__init__(subjectID)
        self.MEG_config = 'MEGconfig_Cb'
        self.MRI_config = 'MRIconfig_C5'
        self.cv_descriptor = None
        self.configure_paths() 
        self.configure_cv()