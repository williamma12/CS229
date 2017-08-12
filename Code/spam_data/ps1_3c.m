
numTrainDocs = [50, 100, 200, 400, 800, 1400];

% Create file name
for num = numTrainDocs
    filename = strcat('MATRIX.TRAIN.', int2str(num));

    % Train NB Classifier
    [phi_y, phi_spam, phi_not_spam] = nb_train(filename);

    % Test NB Classifier
    nb_test('MATRIX.TEST', phi_y, phi_spam, phi_not_spam);
end

clear