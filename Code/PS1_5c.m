close all; clear all; clc;

% Load in data
run('load_quasar_data.m');
smoothed = [lambdas];

for i = train_qso'
    % Weighted model fitted with the i-th training example
    X = lambdas;
    Y = train_qso(1,:)';
    tau = 5;

    % Fit weighted model for each value of tau
    y = [];
    for xi = lambdas'
        w = exp(-(xi - X).^2/(2*tau^2));
        W = diag(w, 0);
        theta = inv(X' * W * X) * X' * W * Y;
        y = [y theta*xi];
    end
    smoothed = [smoothed y'];
end

csvwrite('smooth_train_qso.csv', smoothed')

smoothed = [lambdas];

for i = test_qso'
    % Weighted model fitted with the i-th training example
    X = lambdas;
    Y = test_qso(1,:)';
    tau = 5;

    % Fit weighted model for each value of tau
    y = [];
    for xi = lambdas'
        w = exp(-(xi - X).^2/(2*tau^2));
        W = diag(w, 0);
        theta = inv(X' * W * X) * X' * W * Y;
        y = [y theta*xi];
    end
    smoothed = [smoothed y'];
end

csvwrite('smooth_test_qso.csv', smoothed')