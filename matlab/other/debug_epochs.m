% clear all

subjectID = '19840930bigs';
options.inDir = '/home/scc_e_393956/Desktop/reabt/ncc/MEG/epochs_clean2/manual_finish/';
options.pattern = 'maxfilter_True__ica_True__0.5-20Hz__fs_100__[-1.5_1.5]s_detrend_None_clean_meg-epo';
options.outDir = '/home/scc_e_393956/Desktop/reabt/ncc/MEG/matlab_evoked/';

inDir = options.inDir;
pattern = options.pattern;
outDir = options.outDir;

% addpath('/home/scc_e_393956/MATLAB Add-Ons/Collections/FieldTrip')
addpath('/home/scc_e_393956/ncc/rsa/matlab/helpers')
rmpath /opt/matlab/R2026a/toolbox/fixedpoint/fixedpoint/

% ft_defaults()


% automatically find out whether input dataset consists of multiple partial
% files
primaryFile = fullfile(inDir, subjectID, strcat(subjectID, '_', pattern, '.fif'));

cfg = []
cfg.dataset = primaryFile;
cfg.demean          = 'yes';
cfg.baselinewindow  = [-0.2 0];
data=ft_preprocessing(cfg);
% hdr2 = ft_read_header(cfg.dataset);