% Perform measurements on ARFF datasets
clc
clear

% Setting
folds = 10;
conn = database('cv','jan','',...
                'Vendor','PostGreSQL',...
                'Server','127.0.0.1');    
directory = dir('/Users/jan/Documents/Git/cv/data/arff/*.arff');
note = "permutated"
seed = 2001;

for file_name = {directory.name}
    dataset_name = strrep(file_name{1}, '.arff', '')
    arff_file_path = strcat(directory(1).folder, '/', dataset_name, '.arff');
    xml_file_path = strcat(directory(1).folder, '/', dataset_name, '.xml');
    
    [data,featureNames,~,stringVals,relationName] = weka2matlab(loadARFF(arff_file_path));
    
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

    % Create folds
    tic;
    [assignment, dummy, result, indices, grouped, model] = ilp_slack5(y, folds);
    partition = reconstruct_index(assignment, indices)
    runtime = toc;

    % Store the folds
    csvwrite(['/Users/jan/Documents/Git/cv/folds/ilp5/', dataset_name, '.csv'], partition);
    csvwrite(['/Users/jan/Documents/Git/cv/data/csv/', dataset_name, '.y'], y);
    csvwrite(['/Users/jan/Documents/Git/cv/data/csv/', dataset_name, '.x'], x);

    % Log the result
    logger = table(string(dataset_name), "ilp5", runtime, datetime, folds, seed, note, string(mat2str(round(result.x'))), result.objval, result.mipgap, result.nodecount, size(grouped, 1), size(y, 2), 'VariableNames', {'dataset', 'algorithm', 'runtime', 'timestamp', 'folds', 'seed', 'note', 'x', 'objval', 'mipgap', 'node_cnt', 'group_cnt', 'label_cnt'});
    sqlwrite(conn, 'cv_arff', logger, 'catalog', '"cv"', 'schema', 'hp')
end

close(conn)
