function [error] = nb_test(filename, phi_y, phi_spam, phi_not_spam)

[spmatrix, tokenlist, category] = readMatrix(filename);

testMatrix = full(spmatrix);
numTestDocs = size(testMatrix, 1);
numTokens = size(testMatrix, 2);

% Assume nb_train.m has just been executed, and all the parameters computed/needed
% by your classifier are in memory through that execution. You can also assume 
% that the columns in the test set are arranged in exactly the same way as for the
% training set (i.e., the j-th column represents the same token in the test data 
% matrix as in the original training data matrix).

% Write code below to classify each document in the test set (ie, each row
% in the current document word matrix) as 1 for SPAM and 0 for NON-SPAM.

% Construct the (numTestDocs x 1) vector 'output' such that the i-th entry 
% of this vector is the predicted class (1/0) for the i-th  email (i-th row 
% in testMatrix) in the test set.
output = zeros(numTestDocs, 1);

%---------------
% YOUR CODE HERE
for i = 1:numTestDocs
    prob_spam = log(phi_y);
    prob_not_spam = log(1 - phi_y);
    for j = 1:numTokens
        curr_val = testMatrix(i, j);
        prob_spam = prob_spam + curr_val * log(phi_spam(j));
        prob_not_spam = prob_not_spam + curr_val * log(phi_not_spam(j)); 
    end
    if prob_spam >= prob_not_spam
        output(i) = 1;
    else
        output(i) = 0;
    end
end
%---------------


% Compute the error on the test set
y = full(category);
y = y(:);
error = sum(y ~= output) / numTestDocs;

%Print out the classification error on the test set
fprintf(1, 'Test error: %1.4f\n', error);



