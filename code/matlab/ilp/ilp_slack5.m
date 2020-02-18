% Assignment into folds in cross-validation via integer linear programing.
% 
% ASSIGNMENT = ILP(DATA, FOLDS) takes a numerical matrix DATA, 
% where the last column is the label, and the count of desired FOLDS.
% 
% ASSIGNMENT = ILP(DATA, FOLDS, PREVIOUS_X, PREVIOUS_OBJ, PREVIOUS_GAP)
% warm starts the search.
% 
% The distribution over:
%   1) labels
%   2) two-way interactions between the labels
%   3) the powerset of the labels
% is maintained approximatelly equal across all the folds.
% 
% This is a formulation, where we also minimize the count of folds, where some
% of the powersets == 0. In matlabs notation we minimize sum(min(X)==0),
% where X is the assignment of the powersets into folds.
% In other words, we minimize FLCZ.
% 
% ILP is formulated as:
%   Aeq*x == beq
%   A*x <= b
%   min(f*x)
% 
% The feasibility relaxation is performed with slack variables (it is 
% similar to extension of a hard-margin SVM to soft-margin SVM).

% Example:
%   ilp(data, folds)

function [assignment, dummy, result, indices, grouped, model] = ilp_slack5(data, folds, previous_result_x, previous_objective, previous_gap, previous_violating_column_pairs)
% Argument validation
validateattributes(data, {'numeric'}, {'2d'})
validateattributes(folds, {'numeric'}, {'scalar', 'nonnan', 'finite', 'positive'})

% Initialization
nrow = size(data, 1);
ncol = size(data, 2);

% Grouping (this sorts the rows)
[grouped, cnt, indices] = group_index(data);

% Dummy encoding
dummy = classreg.regr.modelutils.designmatrix(grouped, 'PredictorVars', 1:ncol, 'CategoricalVars', true(ncol,1), 'Intercept', false, 'DummyVarCoding', 'full'); 

% Count of rows/columns after grouping & dummy encoding
grow = size(dummy,1);     
gcol = size(dummy,2);

% Count of unique pairwise interactions
ideal2row = gcol*(gcol-1)/2; 

% Ideal 
ideal1 = (cnt'*dummy)/folds;    % Marginal probability
idealn = cnt/folds;             % Joint probability


%%% Optization formulation
% The constraint matrices (A and Aeq) must be sparse to avoid running out of memory.
% The matrices must be of double type (because Gurobi does not accept indices).

% Initial solution
x0 = sparse(grow, folds);


% Each sample must be used exactly cnt times
Aeq = [repmat(speye(grow), 1, folds), sparse(grow, gcol), sparse(grow, ideal2row)];
beq = cnt;


% Each unique row can be assigned into a fold at most ceil(idealn) times 
% and at least floor(idealn) times. Both bounds are needed.
A = [speye(grow*folds), sparse(grow*folds, gcol), sparse(grow*folds, ideal2row)];
b = repmat(ceil(idealn), folds, 1);

A = [A; -speye(grow*folds), sparse(grow*folds, gcol), sparse(grow*folds, ideal2row)];
b = [b; -repmat(floor(idealn), folds, 1)];


% Two-way interactions
btmp = zeros(ideal2row, 1); 
all_i = nan(ideal2row, 1);
all_j = nan(ideal2row, 1);
row = 1;
for i=1:gcol
    for j=i+1:gcol
        all_i(row) = i;
        all_j(row) = j;
        btmp(row) = all(dummy(:,[i,j]),2)' * cnt / folds;
        row = row+1;
    end
end

Atmp = sparse(and(dummy(:,all_i), dummy(:,all_j))');
tmpCell = repmat({Atmp}, 1, folds);

A = [A; blkdiag(tmpCell{:}), sparse(ideal2row*folds, gcol), repmat(-speye(ideal2row), folds, 1)];
b = [b; repmat(ceil(btmp), folds, 1)];

A = [A; -blkdiag(tmpCell{:}), sparse(ideal2row*folds, gcol), repmat(-speye(ideal2row), folds, 1)];
b = [b; -repmat(floor(btmp), folds, 1)];


% Each feature_value can be assigned into a fold ideal1-1 <= x <= ideal1+1 times (max diff=2). Reasoning: It is tough for models to make a sound prediction for a value that they have never observed during the training phase. Some implementations can deal with new values gracefully, while other implementations return an error. This constraint makes sure that whenever possible, at least one sample with the given value is present in the fold. An alternative is to maximize the count of unique values in a fold - while this is undoable in ILP, this would be a natural goal in Constraint Programming. For label, this makes sure that the apriory probability is estimated in each fold to be as close to the sample mean on the whole data set, as possible.
tmpCell = repmat({dummy'}, 1, folds);
A = [A; blkdiag(tmpCell{:}), repmat(-speye(gcol), folds, 1), sparse(gcol*folds, ideal2row)]; % Some slack variables are used here
b = [b; repmat(ceil(ideal1'), folds, 1)];

A = [A; -blkdiag(tmpCell{:}), repmat(-speye(gcol), folds, 1), sparse(gcol*folds, ideal2row)];
b = [b; -repmat(floor(ideal1'), folds, 1)];


% The folds must be of approximately equal size (max diff=1). Reasoning: If we want to mimize variance of AUC in each fold, then due to existence of learning curves (dependence of AUC based on sample size), it is natural to prefer folds of the same size. Note: If we actually need folds of different sizes, it is trivial to set the constraints AND take into account the relative fold size in the previous 2 constraints (a multiplicative constant for each fold).
tmpCell = repmat({ones(1, grow)}, 1, folds);
A = [A; [blkdiag(tmpCell{:}), sparse(folds, gcol), sparse(folds, ideal2row)]];
b = [b; repmat(ceil(nrow/folds), folds, 1)];

A = [A; [-blkdiag(tmpCell{:}), sparse(folds, gcol), sparse(folds, ideal2row)]];
b = [b; -repmat(floor(nrow/folds), folds, 1)];


% Minimize count of folds with at least one zero powerset (FLCZ)
Atmp = -speye(grow*folds); 
A2tmp = zeros(grow*folds, folds); % Slack variables: one per fold
for fold=1:folds
    A2tmp((fold-1)*grow+1:fold*grow, fold) = -1;
end

A = [A, sparse(size(A,1), folds)];    % Extend the current A array 
Aeq = [Aeq, sparse(size(Aeq,1), folds)];    % Extend the current Aeq array with new columns
A = [A; Atmp, sparse(grow*folds, gcol), sparse(grow*folds, ideal2row), A2tmp];
b = [b; -ones(grow*folds, 1)];


% Counts are non-negative integers
lb = [zeros(1, folds*grow + gcol + ideal2row), zeros(1, folds)];
intcon = 1:folds*grow;  % The slack variables are binary but they can be left unconstrained


% The objective (minimize the slack variables)
f = [zeros(grow*folds, 1); 10000*ones(gcol, 1); 100*ones(ideal2row, 1); ones(folds, 1)];    % We minimize the count of used slack variables. The priority are order 1 interactions, than order 2 interactions, and finally powerset.


% In order to accelerate the tightening of the lower bound, we may
% provide some provable lower bounds.
% The count of folds with at least one zero powerset (FLCZ) is at least
flcz_lb = folds - min(cnt);
flcz_vector = [sparse(1, grow*folds + gcol + ideal2row), ones(1, folds)];
A = [A; -flcz_vector]; % We know that flcz_vector >= flcz_lb. But we use <=...
b = [b; -flcz_lb];

% Set previous lower bound, if known
if nargin > 2 && isfinite(previous_objective)
    A = [A; [sparse(1, grow*folds),  -10000*ones(1, gcol), -100*ones(1, ideal2row), -ones(1, folds)]];
    b = [b; -(previous_objective * (1 - previous_gap - 0.0001))];    % We leave some tolerance
end


%%% Matlab solver
% options = optimoptions('intlinprog', 'CutGeneration', 'none', 'IntegerPreprocess','none');
% x = linprog(f,A,b,Aeq,beq,lb, [], []);
% x = intlinprog(f,intcon,A,b,Aeq,beq,lb, [], [], options);
% assignment = reshape(x(1:grow*folds), grow, folds)


%%% Gurobi solver
model.A = [A; Aeq];
model.rhs = [b; beq];
model.sense = [repmat('<', size(A,1),1); repmat('=', size(Aeq,1),1)]; % the constraint sense vector 
model.lb = lb;
model.obj = reshape(f, numel(f), 1);
model.vtype = [repmat('I', 1, grow*folds), repmat('B', 1, gcol + ideal2row + folds)]; % variables are integers, slacks are binary.
model.modelsense = 'min'; % Matlab minimizes -> we minimize here as well

% Attempt to load previous solution
if nargin > 2
    model.start = previous_result_x;
end

% Set some constraints as lazy (provides significant speed up)
model.lazy = [
    zeros(grow*folds, 1);       % n-way interactions ub (important to not be completely lazy)
    ones(grow*folds, 1);        % n-way interactions lb
    ones(ideal2row*folds, 1);   % 2-way interactions ub (there is too many of them, it also speeds up optimization)
    ones(ideal2row*folds, 1);   % 2-way interactions lb
    zeros(gcol*folds, 1);       % 1-way interactions ub (important to not be completely lazy)
    ones(gcol*folds, 1);        % 1-way interactions lb
    zeros(folds, 1);            % fold size ub (25% is active -> better all active)
    zeros(folds, 1);            % fold size lb
    ones(grow*folds, 1);        % FLCZ
    0;                          % calculated lb on FLCZ
    zeros(grow, 1)              % Aeq - each sample must be assigned exactly once. This is the most constraining constraint (40% of the constraints is active)
];
if nargin > 2
    model.lazy = [model.lazy; 0];    % LB from previous run
end

% Release memory
clear A
clear b
clear A2tmp
clear Aeq
clear Atmp

% Run with the provided parameters
params.timelimit = 20000;       % In seconds
params.NodefileStart = 0.5;     % If necessary, store the explored nodes on the disk
result = gurobi(model, params);

% We ignore slack variables in the solution
assignment = reshape(result.x(1:grow*folds), grow, folds); 

% ILP actually returns continuous result. With the default accuracy of the 
% solvers, we are safe to just round the result.
assignment = round(assignment);
