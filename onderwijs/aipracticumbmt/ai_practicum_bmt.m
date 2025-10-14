%% ai_practicum_bmt.m 
% Compare Linear Regression and Random Forest for cholesterol prediction 
 
clear; clc; close all; 

%% 1) Load data 

url = 'https://wptmdoorn.github.io/data/FTIR%20data_train_studenten.csv'; 
T = readtable(url, 'Delimiter', ';', 'PreserveVariableNames', true); 

labelVar = 'cholesterol_mmol_L'; 

if ~ismember(labelVar, T.Properties.VariableNames) 
    error('Label variable "%s" not found in the dataset.', labelVar); 
end 

%% 2) Prepare predictors (numeric only) 
Xtable = T(:, vartype('numeric')); 

if ismember(labelVar, Xtable.Properties.VariableNames) 
    Xtable(:, labelVar) = []; 
end 
y = T.(labelVar); 

% Remove rows with missing values 
valid = ~ismissing(y) & all(~ismissing(Xtable), 2); 
Xtable = Xtable(valid, :); 
y = y(valid); 
X = table2array(Xtable); 
featureNames = Xtable.Properties.VariableNames; 

%% 3) Train/test split — classic 80/20 (manual)
rng default
n = size(X, 1);
idx = randperm(n);
ntr = round(0.8 * n);

idxTr = idx(1:ntr);
idxTe = idx(ntr+1:end);

Xtr = X(idxTr, :);  Xte = X(idxTe, :);
ytr = y(idxTr);     yte = y(idxTe);

%% 3b) Print dimensions
fprintf('Total samples: %d | Features: %d\n', n, size(X,2));
fprintf('Train set: %d x %d (Xtr) | %d (ytr)\n', size(Xtr,1), size(Xtr,2), numel(ytr));
fprintf('Test  set: %d x %d (Xte) | %d (yte)\n', size(Xte,1), size(Xte,2), numel(yte));

%% 3c) Print heads of tables
disp('--- Head of original table T ---');
disp(head(T));

disp('--- Head of numeric predictor table Xtable ---');
disp(head(Xtable));

disp('--- Head of training predictors (as table) ---');
disp(head(array2table(Xtr, 'VariableNames', featureNames)));

disp('--- Head of test predictors (as table) ---');
disp(head(array2table(Xte, 'VariableNames', featureNames)));

disp('--- Head of training response ytr ---');
disp(head(table(ytr, 'VariableNames', {labelVar})));

disp('--- Head of test response yte ---');
disp(head(table(yte, 'VariableNames', {labelVar})));
