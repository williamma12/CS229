
% Read in token list
filename = 'TOKENS_LIST';
delimiter = ' ';
formatSpec = '%f%s%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true);
fclose(fileID);
tokens = dataArray{:, 2};
clearvars filename delimiter formatSpec fileID dataArray ans;

% Train NB Classifier
[phi_y, phi_spam, phi_not_spam] = nb_train('MATRIX.TRAIN');

% Calculate predictive power of tokens
ratio = phi_spam ./ phi_not_spam;
[sortedRatio,sortingIndices] = sort(ratio,'descend');
topFiveTokens = tokens(sortingIndices(1:5))

clear