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
Theta1Grad = zeros(size(Theta1));
Theta2Grad = zeros(size(Theta2));

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

a1 = [ones(m, 1), X]; % 5000 x 401
z2 = a1 * Theta1';
a2 = [ones(m, 1), sigmoid(z2)];
z3 = a2 * Theta2';
hypothesisTheta = sigmoid(z3);

for k = 1:num_labels
    yk = y == k; % all elements in matrix y that are of value k
    hthetak = hypothesisTheta(:, k);
    JForK = 1 / m * sum(-yk .* log(hthetak) - (1 - yk) .* log(1 - hthetak));
    J = J + JForK;
end

regularize = lambda / (2 * m) * (sum(sum(Theta1(:, 2:end) .^ 2)) ...
    + sum(sum(Theta2(:, 2:end) .^ 2)));
J = J + regularize;

for t = 1:m
    for k = 1:num_labels
        yk = y(t) == k;
        delta3(k) = hypothesisTheta(t, k) - yk;
    end
    delta2 = Theta2' * delta3' .* sigmoidGradient([1, z2(t, :)])';
    delta2 = delta2(2:end);

    Theta1Grad = Theta1Grad + delta2 * a1(t, :);
    Theta2Grad = Theta2Grad + delta3' * a2(t, :);
end

Theta1Grad = Theta1Grad / m;
Theta2Grad = Theta2Grad / m;


Theta1Grad(:, 2:end) = Theta1Grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2Grad(:, 2:end) = Theta2Grad(:, 2:end) + lambda / m * Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1Grad(:) ; Theta2Grad(:)];


end
