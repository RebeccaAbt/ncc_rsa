clear
clc

addpath('/home/reabt/Matlab_toolboxes/spm12/')
spm('Defaults','fMRI');
spm_jobman('initcfg');

% Paths
first_level_base = '/home/reabt/mnt/data/MRI/neurokog/NCC25/analyze_fin';
first_level_rel  = fullfile('NCC','firstLevel_supraSens_M4B');
png_outDir = '/home/reabt/experiments/ncc/MRI/code/matlab_plots';

% Which first-level contrast to inspect
con_idx  = 1;
% con_name = 'tac_hit_vs_miss';

 contrast_names = {'aud_hit_vs_miss'
    'tac_hit_vs_miss'
    'vis_hit_vs_miss'
'NT_hit_vs_miss'};

 con_name = contrast_names{con_idx};

%% Threshold settings
% 'none' = uncorrected
% 'FWE'  = voxelwise FWE-corrected
% 'FDR'  = voxelwise FDR-corrected
thresh_desc = 'FWE';

% p-threshold
thresh_p = 0.05;

% cluster extent threshold
extent_k = 0;

%% Save settings
save_figures = true;
output_dir = '/home/reabt/experiments/ncc/MRI/data/first_level_thresholded_tactile';

if save_figures && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Subject list
allFolders = dir(first_level_base);
allFolders = allFolders([allFolders.isdir]);
allFolders = allFolders(~ismember({allFolders.name}, {'.','..'}));

names = {allFolders.name};
isSubject = ~cellfun('isempty', regexp(names, '^\d{8}[A-Za-z]{4}$'));
subjects = names(isSubject);

fprintf('Found %d subject folders.\n', numel(subjects));

%% Loop over subjects
for s = 1:numel(subjects)

    subj = subjects{s};
    subj_dir = fullfile(first_level_base, subj, first_level_rel);
    spm_mat  = fullfile(subj_dir, 'SPM.mat');

    fprintf('\n============================================================\n');
    fprintf('Subject %d/%d: %s\n', s, numel(subjects), subj);
    fprintf('SPM.mat: %s\n', spm_mat);
    fprintf('============================================================\n');

    if ~exist(spm_mat, 'file')
        warning('SPM.mat not found for subject %s', subj);
        continue
    end

    try
        load(spm_mat, 'SPM');

        if ~isfield(SPM, 'xCon') || numel(SPM.xCon) < con_idx
            warning('Contrast %d not found for subject %s', con_idx, subj);
            continue
        end

        spm_figure('GetWin', 'Graphics');
        spm_figure('Clear', 'Graphics');

        xSPM = struct();
        xSPM.swd       = subj_dir;
        xSPM.Ic        = con_idx;
        xSPM.Im        = [];
        xSPM.pm        = [];
        xSPM.Ex        = [];
        xSPM.title     = sprintf('%s | %s', subj, SPM.xCon(con_idx).name);
        xSPM.thresDesc = thresh_desc;
        xSPM.u         = thresh_p;
        xSPM.k         = extent_k;
        xSPM.units     = {'mm' 'mm' 'mm'};

        [hReg, xSPM, SPM] = spm_results_ui('Setup', xSPM);

        try
            spm_sections(xSPM, hReg, fullfile(spm('Dir'), 'canonical', 'single_subj_T1.nii'));
        catch
            warning('Could not draw sections for subject %s', subj);
        end

        drawnow;

        if save_figures
            figfile = fullfile(png_outDir, sprintf('con_%04d_%s_%s_%s_k_%d_%s.png', con_idx, con_name, num2str(thresh_desc), num2str(thresh_p), extent_k, subj));
            % figfile = fullfile(output_dir, sprintf('%s_con_%04d_%s_%s_p%s_k%d.png', ...
                % subj, con_idx, con_name, thresh_desc, strrep(num2str(thresh_p),'.','p'), extent_k));
            saveas(spm_figure('FindWin', 'Graphics'), figfile);
            fprintf('Saved figure: %s\n', figfile);
        end

    catch ME
        warning('Failed for subject %s: %s', subj, ME.message);
    end
end

fprintf('\nFinished plotting thresholded first-level maps.\n');