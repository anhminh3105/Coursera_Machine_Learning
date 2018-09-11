function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ ones(m,1) X]; % add the bias column to the data set m x (n+1)
a1 = X'; % transpose data set to dimension (n_a1+1) x m
z2 = Theta1*a1; % (n_a2 x m) = (n_a2 * (n_a1+1)) * ((n_a1+1) x m)
a2 = sigmoid(z2);
a2 = [ones(1, m); a2]; % add the bias unit to the hidden layer ((n_a2)+1) x m
z3 = Theta2*a2; % (n_a3 x m) = ( (n_a3 x (n_a2+1)) * ((n_a2+1)x m) )
a3 = sigmoid(z3);

y = sparse(1:m, y, 1); % m x num_labels where n_a3 == num_labels

Theta1_reg = Theta1;
Theta1_reg(:,1) = 0;

Theta2_reg = Theta2;
Theta2_reg(:, 1) = 0;

J = -1/m * sum(sum(y'.*log(a3) + (1-y)' .* log(1-a3))) + lambda/(2*m) * (sum(sum(Theta1_reg.^2)) + sum(sum(Theta2_reg.^2)));

Delta3 = a3 - y'; % n_a3 x m
Delta2 = (Theta2' * Delta3) .* a2 .* (1-a2); % (n_a2+1) x m

DELTA2 = Delta3 * a2'; % n_a3 x (n_a2+1)
DELTA1 = Delta2 * X; % (n_a2+1) x (n_a1+1)  as X == a1'

Theta1_grad = 1/m * (DELTA1(2:end,:) + lambda*Theta1_reg);
Theta2_grad = 1/m * (DELTA2 + lambda*Theta2_reg);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
