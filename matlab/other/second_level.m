clear
clc

addpath('/home/reabt/Matlab_toolboxes/spm12/')

spm('Defaults','fMRI');
spm_jobman('initcfg');



%%
% Path that contains one first-level folder per subject
% Example:
% /data/teaching/fMRI_SE/NCC25/analyze_fin/<subject>/NCC/firstLevel_3x2/
first_level_base = '/home/reabt/mnt/data/MRI/neurokog/NCC25/analyze_fin';

% Output base directory for second-level analyses
second_level_base = '/home/reabt/experiments/ncc/MRI/data/second_level_M4b';
% Name of the first-level folder relative to each subject
first_level_rel = fullfile('NCC','firstLevel_supraSens_M4B');


folderIdx = [];
allFolders = dir(first_level_base);
allFolders = allFolders([allFolders.isdir]); % keep only directories
allFolders = allFolders(~ismember({allFolders.name}, {'.','..'}));

names = {allFolders.name};
idx = cellfun(@length, names) == 12;

subjects = names(idx);
%%

% subjects = subjects([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]); % Version A--> all subjects with SPM.mat
% prefix = 'ver_A';

subjects = subjects([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 19, 20, 21, 22, 25, 27]); % Version B - without 5, 18, 23, 24, 26, 
prefix = 'ver_B';

second_level_base = strcat(second_level_base, '_', prefix);

% Your first-level contrast numbers and names
% Example only: adapt to your real con numbers / labels
contrast_numbers = [1 2 3 4 6 7 8];
contrast_names = {'aud_hit_vs_miss', ...
    'tac_hit_vs_miss', ...
    'vis_hit_vs_miss', ...
    'NT_hit_vs_miss', ...
    'aud_hit+NT_vs_bl', ...
    'tac_hit+NT_vs_bl', ...
    'vis_hit+NT_vs_bl'};

% Optional: delete existing second-level folders before rerunning
overwrite_existing = true;

%% ------------------------------------------------------------------------
% RUN SECOND-LEVEL ANALYSIS FOR EACH CONTRAST
% -------------------------------------------------------------------------

for c = 1:numel(contrast_numbers)

    con_nr   = contrast_numbers(c);
    con_name = contrast_names{c};

    fprintf('\n============================================================\n');
    fprintf('Running second-level analysis for contrast %d: %s\n', con_nr, con_name);
    fprintf('============================================================\n');

    % Output directory for this contrast
    outdir = fullfile(second_level_base, sprintf('con_%04d_%s', con_nr, con_name));

    if overwrite_existing && exist(outdir, 'dir')
        fprintf('Deleting existing folder: %s\n', outdir);
        rmdir(outdir, 's');
    end

    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end

    % Collect first-level contrast images
    scans = cell(numel(subjects), 1);
    missing_subjects = {};

    for s = 1:numel(subjects)
        subj = subjects{s};

        con_file = fullfile(first_level_base, subj, first_level_rel, sprintf('con_%04d.nii,1', con_nr));

        if exist(strtok(con_file, ','), 'file')
            scans{s} = con_file;
        else
            missing_subjects{end+1} = subj; %#ok<SAGROW>
        end
    end

    % Remove missing entries
    scans = scans(~cellfun('isempty', scans));

    if isempty(scans)
        warning('No scans found for contrast %d (%s). Skipping.', con_nr, con_name);
        continue
    end

    if ~isempty(missing_subjects)
        fprintf('Warning: missing contrast image for these subjects:\n');
        disp(missing_subjects')
    end

    fprintf('Found %d subject contrast images.\n', numel(scans));

    % Clear batch
    matlabbatch = {};

    %% --------------------------------------------------------------------
    % 1) Specify second-level one-sample t-test
    % ---------------------------------------------------------------------
    matlabbatch{1}.spm.stats.factorial_design.dir = {outdir};
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = scans;
    matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

    %% --------------------------------------------------------------------
    % 2) Estimate model
    % ---------------------------------------------------------------------
    matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(outdir, 'SPM.mat')};
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

    %% --------------------------------------------------------------------
    % 3) Define second-level contrasts
    % ---------------------------------------------------------------------
    matlabbatch{3}.spm.stats.con.spmmat = {fullfile(outdir, 'SPM.mat')};

    % Positive effect across subjects
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = [con_name ' positive'];
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

    % Negative effect across subjects
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = [con_name ' negative'];
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = -1;
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

    matlabbatch{3}.spm.stats.con.delete = 1;

    %% --------------------------------------------------------------------
    % 4) Run batch
    % ---------------------------------------------------------------------
    spm_jobman('run', matlabbatch);

    fprintf('Finished second-level analysis for: %s\n', con_name);
end

fprintf('\nAll requested second-level analyses finished.\n');