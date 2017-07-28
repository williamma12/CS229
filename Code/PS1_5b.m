close all; clear all; clc;

% Load in data
run('load_quasar_data.m');

% Non-weighted model fitted with the first training example
X = lambdas;
Y = train_qso(1,:)';
theta = inv(X' * X) * X' * Y;

% Plot non-weighted model and corresponding points
figure; hold on;
plot(X, Y, 'go', 'linewidth', 2);
x1 = min(X):1:max(X);
x2 = theta * x1;
plot(x1, x2, 'linewidth', 2);
xlabel('lambda');
ylabel('flux');

% Weighted model fitted with the first training example
X = lambdas;
Y = train_qso(1,:)';
% taus = [5];
taus = [1, 10, 100, 1000];

% Make plot for weighted model
figure; hold on;
plot(lambdas, Y, 'kx', 'linewidth', 2);
x1 = min(lambdas):1:max(lambdas);
xlabel('lambda');
ylabel('flux');

% Fit weighted model for each value of tau
for tau = taus   
    y = [];
    for xi = lambdas'
        w = exp(-(xi - X).^2/(2*tau^2));
        W = diag(w, 0);
        theta = inv(X' * W * X) * X' * W * Y;
        y = [y theta*xi];
    end
x2 = y';
plot(x1, x2, 'linewidth', 2);
end
