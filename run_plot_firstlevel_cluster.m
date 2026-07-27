
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: run_plot_firstlevel_cluster.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clc

cd /home/reabt/experiments/ncc/MRI/code/

% If your plus_slurm setup needs this, keep it:
addpath(genpath('/home/reabt/Matlab_toolboxes/obob_ownft/'))
cfg0 = [];
cfg0.package.plus_slurm = true;
cfg0.verbose = false;
obob_init_ft(cfg0);

% paths need to be added AFTER obob init
addpath('/home/reabt/Matlab_toolboxes/plus_slurm_matlab')
addpath('/home/reabt/experiments/ncc/MRI/code/clusterjobs')


% -------------------------------------------------------------------------
% User settings
% -------------------------------------------------------------------------
first_level_base = '/home/reabt/mnt/data/MRI/neurokog/NCC25/analyze_fin';
con_idx = {6, 7, 8};   % 1=aud, 2=tac, 3=vis, 4=NT

% -------------------------------------------------------------------------
% Collect subject list
% -------------------------------------------------------------------------
allFolders = dir(first_level_base);
allFolders = allFolders([allFolders.isdir]);
allFolders = allFolders(~ismember({allFolders.name}, {'.','..'}));

names = {allFolders.name};
isSubject = ~cellfun('isempty', regexp(names, '^\d{8}[A-Za-z]{4}$'));
subjects = names(isSubject);

fprintf('Found %d subjects.\n', numel(subjects));

% Optional: only submit subjects that actually have SPM.mat
first_level_rel = fullfile('NCC','firstLevel_supraSens_M4B');
keep = false(size(subjects));

for i = 1:numel(subjects)
    spm_mat = fullfile(first_level_base, subjects{i}, first_level_rel, 'SPM.mat');
    keep(i) = exist(spm_mat, 'file') == 2;
end

subjects = subjects(keep);
%%

% subjects = subjects([4, 12, 14, 18]);

fprintf('Submitting %d subjects with existing SPM.mat files.\n', numel(subjects));

if isempty(subjects)
    error('No valid subjects found.');
end

% -------------------------------------------------------------------------
% Create slurm job configuration
% -------------------------------------------------------------------------
cfg = [];
cfg.mem = '32G';
cfg.request_time = 60;   % minutes
cfg.qos = 'high_prio';
cfg.java = true; % we need java for plotting
% cfg.exclude_nodes = 'node09.scc-pilot.plus.ac.at,node10.scc-pilot.plus.ac.at';
slurm_struct = plus_slurm.create(cfg);

myJob = plus_slurm.addjob_cell(slurm_struct, 'C_PlotFirstLevelContrast', subjects, con_idx);

% Submit
plus_slurm.submit(myJob)