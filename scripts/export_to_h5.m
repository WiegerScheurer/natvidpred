disp('Starting batch conversion...');
indir = '/project/3018078.02/MEG_ingmar/';
mat_files = dir(fullfile(indir, 'sub*_100Hz_badmuscle_badlowfreq_badcomp.mat'));

for k = 1:length(mat_files)
    infile = fullfile(indir, mat_files(k).name);
    [~, name, ~] = fileparts(mat_files(k).name);
    outfile = fullfile(indir, [name '_v7_3.mat']);
    
    if exist(outfile, 'file')
        disp(['Skipping (already exists): ' outfile]);
        continue;
    end
    
    disp(['Loading: ' infile]);
    data = load(infile);
    disp('Loaded.');
    
    disp(['Saving as .mat v7.3: ' outfile]);
    save(outfile, '-struct', 'data', '-v7.3');
    disp('Finished saving .mat v7.3.');
    
    clear data
    disp('Cleared memory.');
end

disp('Batch conversion complete.');