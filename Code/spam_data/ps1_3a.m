% Train NB Classifier
[phi_y, phi_spam, phi_not_spam] = nb_train('MATRIX.TRAIN');

% Test NB Classifier
nb_test('MATRIX.TEST', phi_y, phi_spam, phi_not_spam);

clear