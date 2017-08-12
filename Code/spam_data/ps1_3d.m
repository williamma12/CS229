numTrainDocs = [50, 100, 200, 400, 800, 1400];

% Create file name
for num = numTrainDocs
    % Train NB Classifier
    [average_alpha, Xtrain, squared_X_train, num_train] = svm_train(num);

    % Test NB Classifier
    svm_test(average_alpha, Xtrain, squared_X_train, num);
end

clear