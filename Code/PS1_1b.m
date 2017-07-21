close all; clear all; clc;

% Read in data
X = load('logistic_x.txt');
Y = load('logistic_y.txt');

% Prepare for fitting
X = [ones(size(X, 1), 1) X];
theta = log_reg(X ,Y, 20);

% Plot
figure; hold on;
plot(X(Y < 0, 2), X(Y < 0, 3), 'rx', 'linewidth', 2);
plot(X(Y > 0, 2), X(Y > 0, 3), 'go', 'linewidth', 2);
x1 = min(X(:,2)):.01:max(X(:,2));
x2 = -(theta(1) / theta(3)) - (theta(2) / theta(3)) * x1;
plot(x1,x2, 'linewidth', 2);
xlabel('x1');
ylabel('x2');


% Logistic regression fitting function
function f = log_reg(X, Y, maxiter)
m = size(X, 1);
n = size(X, 2);
theta = zeros(n, 1);

for i = 1 : maxiter
    expon = Y .* (X * theta);
    h_theta = 1 ./ (1+exp(expon));
    grad = -(1/m) * (X' * (h_theta .* Y));
    H = (1/m) * (X' * diag(h_theta .* (1-h_theta)) * X);
    theta = theta - H \ grad;
end
f = theta;
end