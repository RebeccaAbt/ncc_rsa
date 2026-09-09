%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SPM8 first level statistic batch script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%You need to start matlab before you run this script.
%Type: "matlab2008a".
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% uses the trimmed data!!
% makes conditions definitions separately for each run so we can account for missing conditions in runs
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

clear variables
close all
% ------------------------------------------- find for which subjects the analysis is missing so we don't need to select them manually
baseDir = '/data/neurokog/NCC25/analyze_fin/';
all_dirs = dir(baseDir);
all_dirs([1, 2, end-2:end]) = [];

missing_subj = [];

for i=1:length(all_dirs)
    subj = all_dirs(i).name;
    if ~isdir([baseDir, subj, '/NCC/firstLevel_sensory_M1C'])
        missing_subj = [missing_subj, i];
    end    
end
% ------------------------------------------- 



% Set Directories

% MRI-Server:
mainDir = '/data/neurokog/NCC25/';  % Directorty of experiment
dataDir = strcat(mainDir, 'analyze_fin/');  % subject Folders
onsDir = strcat(mainDir, 'timeStamps/');
scriptDir=strcat(mainDir, 'scripts/');
spmDir = '/opt/software/spm/spm12/';

addpath(genpath('/opt/software/spm/spm12_and_8_toolboxes/DPABI/'));

% Init scripts path & SPM
%--------------------------------------------------------------------------
addpath(scriptDir)
addpath(spmDir);
spm fmri
% spm_defaults
% global defaults

% Specify subject folder names (e.g. v01, v1, 1 etc.)
%----------------------------------------------------

subname = dir(dataDir);
subname = subname([subname(:).isdir]);
subname([1,2],:) = [];
subname = {subname.name};

% we don't use startsub/endsub here. Instead we define subjects an array in the for loop below
% startsub = 1;      % just one Subject for testing Where to start (With respect to the sequence of subject folders in the 'subname' array)
% endsub = 10;

a=''; p='';

for tmpsub = 1:length(subname)
    swdirw{tmpsub} = [dataDir subname{tmpsub} '/NCC/firstLevel_sensory_M1C/'];      % Save SPM.mat to ...
    fdirw{tmpsub} = [dataDir subname{tmpsub} '/NCC/prepro_V1B/'];   % Where the functional files can be found ...
    Onsfile{tmpsub} = [onsDir subname{tmpsub} '_timeStamps_sensory_38.mat'];
end


%Specify your Functionals Filenames
%------------------------------------
filter_S = {'^swaudbFN.*ncc1.*.nii' '^swaudbFN.*ncc2.*S2.nii' '^swaudbFN.*ncc3.*S3.nii' '^swaudbFN.*ncc4.*S4.nii' '^swaudbFN.*ncc5.*S5.nii' '^swaudbFN.*ncc6.*S6.nii'}; %for spm_select: * -> .*.
%%
%--------------------------------------------
% for sub = startsub:endsub %start subject loop

for sub = missing_subj %start subject loop  % better --> more flexible loop
    %--------------------------------------------
    
    load(Onsfile{sub});
    timeStamps = rmfield(timeStamps, 'runNr');
    
    % Specify Design
    %---------------
    nses = 6;
    
    all_conditions = fieldnames(timeStamps)';
    % make conditions definitions separately for each run so we can accound
    % for missing conditions in runs
    
    for ses = 1:nses
        valid_conditions_idx = ~structfun(@isempty, timeStamps(ses));
        valid_conditions = all_conditions(valid_conditions_idx);
        
        cnam{ses} = valid_conditions;
        ncon{ses} = length(cnam{ses});
        
    end
    
    
    % Specify number of scans per Session
    %------------------------------------
    nscans = 854;
    
    
    disp(['Preparing subject ' (subname{sub})]);
    
    % What should be done?
    %--------------------
    run=1; 	%run (or only save) jobs? 1=yes 0 = no
    make_stat = 1; % Specify 1st level SPM.mat
    make_est = 1;  % Estimate SPM.mat - Choose 1 for Classical, 2 for Bayes Estimation (see below)
    make_con = 1;  % Write Contrasts to SPM.mat
    make_scaledcon = 1; %scale contrastimages with alff of residuals
    %-------------------
    
    %---------------
    if make_stat == 1
        %---------------
        
        % Get Functional Files
        
        for ses=1:nses
            tmp{ses} = strcat(fdirw{sub}, spm_select('List',fdirw{sub}, ['^' filter_S{ses}]));
            n = ',1';
            for i = 2:nscans
                m = strcat(',', num2str(i));
                n=strvcat(n, m);
            end
            tmp2{ses} = strcat(tmp{ses}, n);
            matlabbatch{1}.spm.stats.fmri_spec.sess(ses).scans= cellstr(tmp2{ses});
            clear i n m;
        end
        
        % make results directory
        %-----------------------
        eval(sprintf('!mkdir %s',swdirw{sub}));
        cd(swdirw{sub});
        %!cp ../f3fmap/rp* ../f3fmapdartel/;
        
        % Settings: Timing
        %--------------------------
        matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs'; % OPTIONS: 'scans'|'secs' for onsets
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 1.05;%2.25;	  % TR
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;	  % Size of time bins for onset specification
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;	  % Microtime onset
        
        % Settings: Basis Functions
        %--------------------------
        matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
        %By Johannes: struct('name',zeros(0),'levels',zeros(0));
        
        matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];    % Options:
        % [0 0] No Derivatives
        % [1 0] Time Derivatives
        % [1 1] Time and Dispersion Derivatives
        
        matlabbatch{1}.spm.stats.fmri_spec.volt = 1;                    % Options: 1 = no; 2 = yes
        % Set Output Directory
        %---------------------
        matlabbatch{1}.spm.stats.fmri_spec.dir = cellstr(swdirw{sub});
        
        
        % Load onset file
        %----------------
%         cd(onsDir);
        
        
        % Set onset vectors
        %------------------
        
        for ses = 1: nses
            conditions =cnam{ses}';
            for iCondition = 1:ncon{ses}-2  % for all stimuli, but not the last 2 regressors
                matlabbatch{1}.spm.stats.fmri_spec.sess(ses).cond(iCondition).onset = timeStamps(ses).(conditions{iCondition})(:,1);
                matlabbatch{1}.spm.stats.fmri_spec.sess(ses).cond(iCondition).duration = 0;
            end
            
            for iCondition = ncon{ses}-1:ncon{ses} % only for the last 2 regressors: define other durations
                matlabbatch{1}.spm.stats.fmri_spec.sess(ses).cond(iCondition).onset = timeStamps(ses).(conditions{iCondition})(:,1);
                matlabbatch{1}.spm.stats.fmri_spec.sess(ses).cond(iCondition).duration = timeStamps(ses).(conditions{iCondition})(:,2);
            end
        end
        %%
        for ses = 1:nses
            fprintf('\nses = %i ', ses) 
            for c = 1:ncon{ses}
                fprintf('\ncon = %i ; condition = %s ', c, cnam{ses}{c}) 
                % This determines the number of zeros needed
                % for building events in event-related design
                l = length(matlabbatch{1}.spm.stats.fmri_spec.sess(ses).cond(c).onset);
                nzeros(l,1) = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess(ses).cond(c).name = cnam{ses}{c};
                matlabbatch{1}.spm.stats.fmri_spec.sess(ses).cond(c).tmod = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess(ses).cond(c).pmod = struct('name', {}, 'param', {}, 'poly', {});
                %matlabbatch{1}.spm.stats.fmri_spec.sess(ses).cond(c).duration = nzeros;
                clear l nzeros;
            end
        end
        %%
        
        % Trial specification: Onsets, duration (UNITS) and parameters for modulation
        %----------------------------------------------------------------------------
        for ses = 1:nses
            matlabbatch{1}.spm.stats.fmri_spec.sess(ses).multi = {''}; % Trial specification using Text files ==> Not in use
        end
        
        % Realignment Parameters
        %-----------------------
        mpfilter = {'^rp_dbFN.*ncc1.*.txt' '^rp_dbFN.*ncc2.*.txt' '^rp_dbFN.*ncc3.*.txt' '^rp_dbFN.*ncc4.*.txt' '^rp_dbFN.*ncc5.*.txt' '^rp_dbFN.*ncc6.*.txt'};
        
        for ses = 1:nses
            rpfile{ses} = strcat(fdirw{sub}, spm_select('List',fdirw{sub},mpfilter{ses}));
        end
        
        nreg = {'R1' 'R2' 'R3' 'R4' 'R5' 'R6'};
        
        for ses = 1:nses
            for mp = 1:6
                vreg = textread(rpfile{ses});
                matlabbatch{1}.spm.stats.fmri_spec.sess(ses).regress(mp).name = nreg{mp};
                matlabbatch{1}.spm.stats.fmri_spec.sess(ses).regress(mp).val = vreg(1:nscans,mp);
                clear vreg;
            end
        end
        
        % Regressors
        %-----------
        for ses = 1:nses
            matlabbatch{1}.spm.stats.fmri_spec.sess(ses).hpf = 128;                       % high-pass cutoff (secs) [Inf = no filtering]
        end
        
        %matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi_reg = {[fdirw{sub} 'FIACH/S1/noise_basis6_1.txt']};
        %matlabbatch{1}.spm.stats.fmri_spec.sess(2).multi_reg = {[fdirw{sub} 'FIACH/S2/noise_basis6_2.txt']};
        
        % Other settings (Global Normalization, Explicit Masking, intrinsic autocorrelation)
        %-----------------------------------------------------------------------------------
        matlabbatch{1}.spm.stats.fmri_spec.global = 'None';   % Global Normalisation: Choose 'None' or 'Scaling'
        matlabbatch{1}.spm.stats.fmri_spec.mask = {''};       % Not implemented
        matlabbatch{1}.spm.stats.fmri_spec.cvi = 'FAST';     % Serial correlations: Choose 'AR(1)' or 'none'
        
        
        % Build SPM.mat
        %--------------
        cd(swdirw{sub});
        eval(sprintf(['save ' [subname{sub}(1:end-length(p)) '_stat.mat'] ' matlabbatch']));
        if run, spm_jobman('run',matlabbatch); end;
        disp(sprintf('Done.'));
        
        clear tmp matlabbatch;
        
        %-----------------
    end % of make_stat
    %-----------------
    
    
    % Estimation
    %(Bayesian Estimation not implemented)
    %---------------
    if make_est == 1
        %---------------
        
        matlabbatch{1}.spm.stats.fmri_est.spmmat = {[swdirw{sub} 'SPM.mat']};
        matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
        matlabbatch{1}.spm.stats.fmri_est.write_residuals = 1;
        
        cd(swdirw{sub});
        eval(sprintf(['save ' [subname{sub}(1:end-length(p)) '_est.mat'] ' matlabbatch']));
        if run, spm_jobman('run',matlabbatch); end;
        disp(sprintf('Done.'));
        clear matlabbatch;
        
        %-----------------
    end % of make_est
    %-----------------
    
%-------------------
end %of subject loop
%-------------------
cd(scriptDir);

