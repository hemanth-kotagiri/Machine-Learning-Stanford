function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  m = length(y);
  for iter = 1:num_iters,
    error = (X*theta)-y;
    temp0 = theta(1) -(alpha/m)*sum(error.*(X(:,1)));
    temp1 = theta(2) - (alpha/m)* sum(error.*(X(:,2)));
    theta = [temp0; temp1];
    J_history(iter) = computeCost(X, y, theta);
  endfor
endfunction
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
load 'ex1data1.txt'
X = ex1data1;
y = ex1data1(:,2);

num_iters = 40;
alpha = 0.01

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,
    gradientDescent(X,y,theta,alpha,num_iters)
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

%end
