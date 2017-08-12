function [average_alpha, Xtrain, squared_X_train, num_train] = svm_train(num_train)

% Before using this method, set num_train to be the number of training
% examples you wish to read.

[sparseTrainMatrix, tokenlist, trainCategory] = ...
    readMatrix(sprintf('MATRIX.TRAIN.%d', num_train));

% Make y be a vector of +/-1 labels and X be a {0, 1} matrix.
ytrain = (2 * trainCategory - 1)';
Xtrain = 1.0 * (sparseTrainMatrix > 0);

numTrainDocs = size(Xtrain, 1);
numTokens = size(Xtrain, 2);

% Xtrain is a (numTrainDocs x numTokens) sparse matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents if the j-th token appears in
% email i.

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% For the SVM, we convert these to +1 and -1 to form the numTrainDocs x 1
% vector ytrain.

% This vector should be output by this method
average_alpha = zeros(numTrainDocs, 1);

%---------------
% YOUR CODE HERE

tau = 8;
steps = 40 * num_train;
lambda = 1 / (64 * num_train);
alpha = zeros(numTrainDocs, 1);

squared_X_train = sum(Xtrain.^2, 2);
middle_matrix = Xtrain * Xtrain';
K = full(exp(-(repmat(squared_X_train, 1, num_train) + ...
    repmat(squared_X_train', num_train, 1) + 2 * middle_matrix)/ (2 * tau^2)));

for i = 1:steps
    stepsize = 1/sqrt(i);
    index = ceil(rand * num_train);
    margin = ytrain(index) * K(:, index)' * alpha;
    g = -(margin < 1) * ytrain(index)* K(:, index)...
        + lambda * K(:, index) * alpha(index);
    alpha = alpha - stepsize * g;
    average_alpha = average_alpha + alpha;
end

average_alpha = average_alpha ./ steps;
%---------------
