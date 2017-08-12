function [phi_y, phi_spam, phi_not_spam] = nb_train(filename)

[spmatrix, tokenlist, trainCategory] = readMatrix(filename);

trainMatrix = full(spmatrix);
numTrainDocs = size(trainMatrix, 1);
numTokens = size(trainMatrix, 2);

% trainMatrix is now a (numTrainDocs x numTokens) matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents the number of times the j-th
% token appeared in email $i$. 

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% Note that for the SVM, you would want to convert these to +1 and -1.


% YOUR CODE HERE

numSpam = 0;

phi_spam_numerators = zeros(numTokens, 1);
phi_spam_denominators = zeros(numTokens, 1);

phi_not_spam_numerators = zeros(numTokens, 1);
phi_not_spam_denominators = zeros(numTokens, 1);


% Laplace smoothing
phi_spam_numerators(:) = 1;
phi_not_spam_numerators(:) = 1;
phi_spam_denominators(:) = numTokens;
phi_not_spam_denominators(:) = numTokens;

for i = 1:numTrainDocs
    y = trainCategory(i);
    x_i = trainMatrix(i,:);
    if y == 1
        numSpam = numSpam + 1;
    end
    
    docNumWords = 0;
    
    for j = 1:numTokens
        x_ij = x_i(j);
        docNumWords = docNumWords + x_ij;
        if y == 1
            phi_spam_numerators(j) = phi_spam_numerators(j) + x_ij;
        else
            phi_not_spam_numerators(j) = phi_not_spam_numerators(j) + x_ij;
        end
    end
    
    if y == 1
        phi_spam_denominators(:) = phi_spam_denominators(:) + docNumWords;
    else
        phi_not_spam_denominators(:) = phi_not_spam_denominators(:) + docNumWords;
    end
end

phi_y = numSpam / numTrainDocs;
phi_spam = phi_spam_numerators ./ phi_spam_denominators;
phi_not_spam = phi_not_spam_numerators ./ phi_not_spam_denominators;

clearvars numSpam phi_spam_numerators phi_spam_denominators
clearvars phi_not_spam_numerators phi_not_spam_denominators
clearvars docNumWords i j numTrainDocs spmatrix tokenlist x_i x_ij y