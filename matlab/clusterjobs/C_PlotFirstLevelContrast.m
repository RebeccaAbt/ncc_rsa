%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: matlab/C_PlotFirstLevelContrast.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef C_PlotFirstLevelContrast < plus_slurm.slurm_job
    methods
        function run(obj, subj, con_idx)

            clearvars -except subj con_idx

            addpath('/home/reabt/Matlab_toolboxes/spm12/')
            spm('Defaults','fMRI');
            spm_jobman('initcfg');

            % Paths
            first_level_base = '/home/reabt/mnt/data/MRI/neurokog/NCC25/analyze_fin';
            first_level_rel  = fullfile('NCC','firstLevel_supraSens_M4B');
            png_outDir       = '/home/reabt/experiments/ncc/MRI/code/matlab_plots';

            % Contrast names
            contrast_names = { ...
                'aud_hit_vs_miss', ...
                'tac_hit_vs_miss', ...
                'vis_hit_vs_miss', ...
                'NT_hit_vs_miss',...
                'weird_stuff',...
                'aud_hit+NT_vs_bl', ...
                'tac_hit+NT_vs_bl', ...
                'vis_hit+NT_vs_bl' ...
               
                };

            if con_idx < 1 || con_idx > numel(contrast_names)
                error('Invalid con_idx: %d', con_idx);
            end

            con_name = contrast_names{con_idx};

            % Cluster extent threshold
            extent_k = 0;

            % Threshold options
            thresh_desc_list = {'none', 'none', 'FWE', 'FWE'};
            thresh_p_list    = [0.001, 0.01, 0.05, 0.1];

            % Save settings
            save_figures = true;

            if save_figures && ~exist(png_outDir, 'dir')
                mkdir(png_outDir);
            end

            subj_dir = fullfile(first_level_base, subj, first_level_rel);
            spm_mat  = fullfile(subj_dir, 'SPM.mat');

            fprintf('\n============================================================\n');
            fprintf('Subject: %s\n', subj);
            fprintf('Contrast index: %d\n', con_idx);
            fprintf('SPM.mat: %s\n', spm_mat);
            fprintf('============================================================\n');

            if ~exist(spm_mat, 'file')
                error('SPM.mat not found for subject %s:\n%s', subj, spm_mat);
            end

            load(spm_mat, 'SPM');

            if ~isfield(SPM, 'xCon') || numel(SPM.xCon) < con_idx
                error('Contrast %d not found for subject %s', con_idx, subj);
            end

            % Keep an untouched copy in case spm_results_ui modifies SPM
            SPM_loaded = SPM;

            % Loop over threshold settings
            for t = 1:numel(thresh_desc_list)

                thresh_desc = thresh_desc_list{t};
                thresh_p    = thresh_p_list(t);

                fprintf('  Threshold option %d/%d: %s, p = %g, k = %d\n', ...
                    t, numel(thresh_desc_list), thresh_desc, thresh_p, extent_k);

                try
                    SPM = SPM_loaded; %#ok<NASGU>

                    spm_figure('GetWin', 'Graphics');
                    spm_figure('Clear', 'Graphics');

                    xSPM = struct();
                    xSPM.swd       = subj_dir;
                    xSPM.Ic        = con_idx;
                    xSPM.Im        = [];
                    xSPM.pm        = [];
                    xSPM.Ex        = [];
                    xSPM.title     = sprintf('%s | %s | %s p=%g', subj, SPM.xCon(con_idx).name, thresh_desc, thresh_p);
                    xSPM.thresDesc = thresh_desc;
                    xSPM.u         = thresh_p;
                    xSPM.k         = extent_k;
                    xSPM.units     = {'mm' 'mm' 'mm'};

                    [hReg, xSPM, SPM] = spm_results_ui('Setup', xSPM); %#ok<ASGLU>

                    try
                        spm_sections(xSPM, hReg, fullfile(spm('Dir'), 'canonical', 'single_subj_T1.nii'));
                    catch ME_sections
                        warning('Could not draw sections for subject %s (%s, p=%g): %s', ...
                            subj, thresh_desc, thresh_p, ME_sections.message);
                    end

                    drawnow;

                    if save_figures
                        % p_str = strrep(num2str(thresh_p), '.', 'p');
                        figfile = fullfile(png_outDir, ...
                            sprintf('con_%04d_%s_%s_p_%s_k_%d_%s.png', ...
                            con_idx, con_name, thresh_desc, num2str(thresh_p), extent_k, subj));

                        saveas(spm_figure('FindWin', 'Graphics'), figfile);
                        fprintf('    Saved figure: %s\n', figfile);
                    end

                catch ME_thresh
                    warning('Thresholding failed for subject %s (%s, p=%g): %s', ...
                        subj, thresh_desc, thresh_p, ME_thresh.message);
                end
            end

            fprintf('\nFinished plotting thresholded first-level maps for subject %s.\n', subj);
        end
    end
end

