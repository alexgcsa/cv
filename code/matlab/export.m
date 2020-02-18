% Read solutions from the db and write them into csv.
clc
clear

% Setting
setdbprefs('DataReturnFormat', 'table');
conn = database('cv','jan','',...
                'Vendor','PostGreSQL',...
                'Server','127.0.0.1');  
directory = dir('/Users/jan/Documents/Git/cv/data/arff/*.arff');
seed = 98;

% Get assignments from the db
resultset = fetch(conn, 'select t2.dataset, t2.x from hp.cv t1 join hp.cv_arff t2 using(dataset, "timestamp") where dataset not in (''tmc2007'', ''Nuswide_BoW'', ''Nuswide_cVLADplus'')');

% For each assignment, get the folds
for row=1:height(resultset)
    dataset_name = resultset.dataset{row}
    result_x = eval(resultset.x{row});
    
    % Load the original dataset    
    arff_file_path = strcat(directory(1).folder, '/', dataset_name, '.arff');
    xml_file_path = strcat(directory(1).folder, '/', dataset_name, '.xml');
    [data,featureNames,~,stringVals,relationName] = weka2matlab(loadARFF(arff_file_path));
  
    % Parse Mulan XML file
    xml = fileread(xml_file_path);
    labels = regexp(xml, '<label name="([^"]*)">\s*<\/label>', 'tokens');
    
    % Split the data into X and Y
    targets = [];

    for label=labels
        for i = 1:length(featureNames)
            if strcmp(label{:}, featureNames{i})
                targets =[targets; i];
            end
        end
    end

    x = data;
    x(:, targets) = [];
    y = data(:, targets);
    
    
    
    % Grouping (this sorts the rows)
    [grouped, cnt, indices] = group_index(y);

    % Dummy encoding
    ncol = size(y, 2);
    dummy = classreg.regr.modelutils.designmatrix(grouped, 'PredictorVars', 1:ncol, 'CategoricalVars', true(ncol,1), 'Intercept', false, 'DummyVarCoding', 'full'); 

    % Reconstruction
    grow = size(dummy,1);
    folds = 10;
    assignment = round(reshape(result_x(1:grow*folds), grow, folds));   
    partition = reconstruct_index(assignment, indices);               % in order
%     partition = reconstruct_index(assignment, indices, seed, folds, zscore(x)); % DOB-SCV
   
     % Store the folds
%     csvwrite(['/Users/jan/Documents/Git/cv/folds/ilp5_zscore/', dataset_name, '.csv'], partition);
    
    % Random baseline
    rng(seed)
    csvwrite(['/Users/jan/Documents/Git/cv/folds/random/', dataset_name, '.csv'], partition(randperm(size(y, 1))));
    
end