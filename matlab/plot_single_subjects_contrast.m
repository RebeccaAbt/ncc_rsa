clear
clc

addpath('/home/reabt/Matlab_toolboxes/spm12/')
spm('Defaults','fMRI');
spm_jobman('initcfg');

%% Paths
first_level_base = '/home/reabt/mnt/data/MRI/neurokog/NCC25/analyze_fin';
first_level_rel  = fullfile('NCC','firstLevel_supraSens_M4B');
png_outDir = '/home/reabt/experiments/ncc/MRI/code/matlab_plots';

%% Which contrast to inspect
con_nr   = 2;
con_name = 'tac_hit_vs_miss';

%% Optional display settings
save_figures = true;
output_dir   = '/home/reabt/experiments/ncc/MRI/data/first_level_sanitycheck_tactile';

if save_figures && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Get subjects
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
    con_file = fullfile(first_level_base, subj, first_level_rel, sprintf('con_%04d.nii', con_nr));

    fprintf('\n============================================================\n');
    fprintf('Subject %d/%d: %s\n', s, numel(subjects), subj);
    fprintf('Contrast file: %s\n', con_file);
    fprintf('============================================================\n');

    if ~exist(con_file, 'file')
        warning('Contrast file not found for subject %s', subj);
        continue
    end

    % Clear graphics window
    spm_figure('GetWin', 'Graphics');
    spm_figure('Clear', 'Graphics');

    % Display the image
    spm_orthviews('Reset');
    spm_check_registration(char(con_file));

    % Set title in graphics window
    annotation('textbox', [0.01 0.95 0.98 0.04], ...
        'String', sprintf('%s   |   %s   |   con_%04d', subj, con_name, con_nr), ...
        'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 14, ...
        'FontWeight', 'bold');

    drawnow;

    % Save figure
    if save_figures
        figfile = fullfile(png_outDir, sprintf('con_%04d_%s_%s_%s_%s.png', con_nr, con_name, num2str(thresh_desc), num2str(thresh_p), subj));
        saveas(spm_figure('FindWin', 'Graphics'), figfile);
        fprintf('Saved figure: %s\n', figfile);
    end

    % Optional pause so you can inspect manually
    % pause;
end

fprintf('\nFinished plotting all subjects.\n');