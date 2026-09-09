%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SPM12 first level statistic batch script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function eva_firstlevel(subject, files, eventfiles, confoundfiles, contrastsjson, conditionsjson, output_dir)

% Set output dir
%---------------------------------------------------------------------------
if exist(output_dir,'dir')
    rmdir(output_dir,'s');
end
mkdir(output_dir);
cd(output_dir)

diary(fullfile(output_dir,'matlab.log'));
diary on
%-------------------------------------------
% start spm
%-------------------------------------------
addpath /mnt/ceph/groups_hdd/SCCGroup/neurocognition_lab/software/spm12_7771_cat12_r2577; 
spm('Defaults','fMRI');
spm_jobman('initcfg');

%-------------------------------------------
% get input files
%________________
files = split(files, ';');
eventfiles = split(eventfiles, ';');
confoundfiles = split(confoundfiles, ';');

files = cellstr(files);
eventfiles = cellstr(eventfiles);
confoundfiles = cellstr(confoundfiles);

% Load contrast definition
json_txt = fileread(contrastsjson);
contrast_config = jsondecode(json_txt);

contrasts = contrast_config.contrasts;

% Load condition definition
%--------------------------
json_txt = fileread(conditionsjson);
condition_config = jsondecode(json_txt);

%----------------
% Specify Design
%---------------
ncon = length(condition_config.conditions);
nruns = length(files);              %Number of runs
V = spm_vol(files{1});
nscans = numel(V);					%Number of images per file, readout from first input file

% What should be done?
%----------------------------------------------
execute_spm = 1; 	% execute(or only save) jobs? 1=yes 0 = no
make_stat = 1; % Specify 1st level SPM.mat 
make_est = 1;  % Estimate SPM.mat - Choose 1 for Classical, 2 for Bayes Estimation (see below)
make_con = 1;  % Write Contrasts to SPM.mat

%--------------------------
% start actual analyse
%--------------------------

disp(['Preparing subject ' subject]);

%---------------
if make_stat == 1
%---------------
	clear matlabbatch		%to make sure no old variables interfer

	% Get Functional Files
	for run = 1:nruns

		nii_file = files{run};

		V = spm_vol(nii_file);

		scans = cell(length(V),1);

		for v = 1:length(V)
		    scans{v} = sprintf('%s,%d', nii_file, v);
		end

		matlabbatch{1}.spm.stats.fmri_spec.sess(run).scans = scans;

	end

    % Settings: Timing
    %--------------------------
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs'; % OPTIONS: 'scans'|'secs' for onsets
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 1.299;     % TR, usually: 2.25;	
    %matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;	  % Size of time bins for onset specification
    %matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;	  % Microtime onset

    % Settings: Basis Functions
    %--------------------------
    matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});

    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];    % Options: 
                                                                    % [0 0] No Derivatives
                                                                    % [1 0] Time Derivatives
                                                                    % [1 1] Time and Dispersion Derivatives

    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;                    % Options: 1 = no; 2 = yes

    % Set Output Directory
    %---------------------
	matlabbatch{1}.spm.stats.fmri_spec.dir = {output_dir};

    % Set onset vectors
    %------------------
    
    for run = 1:nruns

		ons = spm_load(eventfiles{run});

		for c = 1:ncon
		    
		    condition_name = condition_config.conditions(c).name;
			disp(['condition = ', condition_name]);
		    % start with all events included
		    idx = true(length(ons.onset),1);

			% inclusion criteria
			if isfield(condition_config.conditions(c),'include')

				for r = 1:length(condition_config.conditions(c).include)

					rule = condition_config.conditions(c).include(r);

					column = string(ons.(rule.column));
					value = string(rule.value);

					switch rule.operator

						case 'equals'
						    idx = idx & (column == value);

						case 'not_equals'
						    idx = idx & (column ~= value);

					end

				end

			end
			
			fprintf('Run %d | Condition: %s | %d onsets\n', run, condition_name, sum(idx));
            disp('Onsets = ');
            disp(ons.onset(idx));

		    % hand over to SPM
		    matlabbatch{1}.spm.stats.fmri_spec.sess(run).cond(c).name = condition_name;
		    matlabbatch{1}.spm.stats.fmri_spec.sess(run).cond(c).onset = ons.onset(idx);
		    matlabbatch{1}.spm.stats.fmri_spec.sess(run).cond(c).duration = ons.duration(idx);
			matlabbatch{1}.spm.stats.fmri_spec.sess(run).cond(c).tmod = 0;
		end
		clear ons
	end

    % Realignment Parametersconfound
    %-----------------------
    
    nreg = {'R1' 'R2' 'R3' 'R4' 'R5' 'R6'};
	n_nuisance = 6

    for run = 1:nruns

        % Select confound file for this run
    	confound_file = confoundfiles{run};

    	% Load confounds
    	confounds = spm_load(confound_file);

        vreg = [confounds.trans_x, confounds.trans_y, confounds.trans_z, ...
                confounds.rot_x, confounds.rot_y, confounds.rot_z];
        
        for mp = 1:6
            matlabbatch{1}.spm.stats.fmri_spec.sess(run).regress(mp).name = nreg{mp};
            matlabbatch{1}.spm.stats.fmri_spec.sess(run).regress(mp).val = vreg(1:nscans, mp);
        end
        
        clear confounds vreg;
    end

    % Other settings (Global Normalization, Explicit Masking, intrinsic autocorrelation) 
    %-----------------------------------------------------------------------------------
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';   % Global Normalisation: Choose 'None' or 'Scaling'
    % matlabbatch{1}.spm.stats.fmri_spec.mask = {''};       % Not implemented
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';     % Serial correlations: Choose 'AR(1)' or 'none'  

    % Build SPM.mat
    %--------------
    save(fullfile(output_dir, sprintf('%s_nruns-%d_firstlevel_batch.mat', subject, nruns)), 'matlabbatch');
    if execute_spm, spm_jobman('run',matlabbatch); end;        
    disp(sprintf('Done.'));

    clear tmp matlabbatch;

%-----------------
end % of make_stat
%-----------------

% Estimation
%---------------
if make_est == 1
%---------------

    matlabbatch{1}.spm.stats.fmri_est.spmmat = {fullfile(output_dir,'SPM.mat')};
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
    matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;

	save(fullfile(output_dir, sprintf('%s_nruns-%d_estimation_batch.mat', subject, nruns)), 'matlabbatch');
    if execute_spm
        spm_jobman('run',matlabbatch)
    end
    disp(sprintf('Done.'));
    clear matlabbatch;

%-----------------
end % of make_est
%-----------------

% Contrasts
%---------------
if make_con == 1
%---------------

    % Contrast manager
%-----------------

	disp('Generating contrasts...');

	matlabbatch{1}.spm.stats.con.spmmat = {fullfile(output_dir,'SPM.mat')};
    matlabbatch{1}.spm.stats.con.delete=[1];

	for c = 1:length(contrasts)
		disp(n_nuisance);
		task_vector = contrasts(c).vector(:)';
		full_vector = [task_vector zeros(1,n_nuisance)];
		disp(full_vector);
		disp(length(full_vector));

		matlabbatch{1}.spm.stats.con.consess{c}.tcon.name = contrasts(c).name;
		matlabbatch{1}.spm.stats.con.consess{c}.tcon.convec = full_vector;
		matlabbatch{1}.spm.stats.con.consess{c}.tcon.sessrep = 'repl';

	end

    if execute_spm, spm_jobman('run',matlabbatch); end;
    clear matlabbatch;
    disp('Done.');

%----------------
end % of make_con
%----------------

diary off

end
