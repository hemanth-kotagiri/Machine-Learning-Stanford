function J = computeCost(X, y, theta)
  #data = load('ex1data1.txt'); % read comma separated data
  #X = data(:, 1); y = data(:, 2);
  #theta = 0;
  m = length(y);
  J = sum(((X*theta)-y).^2);
  J = (1/(2*m))*J;
endfunction
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
theta = 0;
m = length(y); % number of training examples


% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.



J(X,y,0);

% =========================================================================

%end
