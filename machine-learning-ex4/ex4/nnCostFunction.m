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

% Problem parameters
%
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
%
% hidden_layer_size 	= 25
% input_layer_size 		= 400
% num_labels 			= 10

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);		% m = rows in X
         
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


% Forward propagation

% Initially X = 5000 x 400
% Adding the bias unit
% X = 5000 x 401

X = [ones(m,1) X];
z2 = Theta1 * X';				% (25 x 401) * (401 x 5000) = (25 x 5000)
activation2 = sigmoid(z2);

% fprintf('\nGot second activation...\n')

activation2 = activation2';			% (5000 x 25)
activation2 = [ones(m,1) activation2];		% (5000 x 26)
z3 = activation2 * Theta2'; 		% (5000 x 26) * (26 x 10) = (5000 x 10)
activation3 = sigmoid(z3);
hTheta = activation3;				% (5000 x 10)

% fprintf('\nGot third activation...\n')

% y = 5000 x 1
% y has values in the range [1,10]
% Need to make a vector, e.g. if y = 5 then must have the row [0 0 0 0 1 0 0 0 0 0]

y_vector = zeros(size(y,1), num_labels);		% (5000 x 10)

for i = 1:size(y,1)
	y_vector(i,y(i)) = 1;
end

J = (1/m) * sum(sum(-y_vector .* log(hTheta) - (1 - y_vector) .* log(1 - hTheta)));
% J = (5000 x 10)

% Removing the first column (bias terms) for regularisation
noBiasTheta1 = Theta1(:,2:size(Theta1,2));
noBiasTheta2 = Theta2(:,2:size(Theta2,2));

regularisation = (lambda/(2*m)) * (sum(sum(noBiasTheta1.^2)) + sum(sum(noBiasTheta2.^2)));

% Cost function, regularised
J = J + regularisation;


% Backpropagation
for t = 1:m
	
	% Step 1
	activation1 = X(t,:);		% (1 x 401)
	activation1 = activation1';	% (401 x 1)
	z2 = Theta1 * activation1;	% (25 x 401) * (401 x 1) = (25 x 1)
	activation2 = sigmoid(z2);	% (25 x 1)

	activation2 = [1; activation2];	% adding bias unit (26 x 1)
	z3 = Theta2 * activation2;		% (10 x 26) * (26 x 1) = (10 x 1)
	activation3 = sigmoid(z3);		% (10 x 1)


	% Step 2
	delta3 = activation3 - y_vector(t,:)';	% (10 x 1)


	% Step 3
	z2 = [1; z2];		% adding bias unit (26 x 1)

	% Theta2 	= (10 x 26)
	% Theta2'	= (26 x 10)
	% delta3	= (10 x 1)
	% 
	% (Theta2' * delta3)	= (26 x 1)
	% % z2 		= (26 x 1)

	delta2 = (Theta2' * delta3) .* sigmoidGradient(z2);		% delta2 = (26 x 1)


	% Step 4
	delta2 = delta2(2:end);		% removing delta2(0)  delta2 = (25 x 1)

	% delta3		= (10 x 1)
	% activation2	= (26 x 1)
	Theta2_grad = Theta2_grad + delta3 * activation2';		% (10 x 26)

	% delta2 		= (25 x 1)
	% activation1	= (401 x 1)
	Theta1_grad = Theta1_grad + delta2 * activation1';		% (25 x 401)

end;


% Step 5
Theta2_grad = (1/m) * Theta2_grad;		% (10 x 26)
Theta1_grad = (1/m) * Theta1_grad;		% (25 x 401)


% Regularisation

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1


% -------------------------------------------------------------

% =========================================================================


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
