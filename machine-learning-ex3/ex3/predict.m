function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% The matrices Theta1 and Theta2 will now be in your Octave
% environment
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

% p -> m x 1 (labels containing 1 to num_labels)
% X -> m x 400
% m = 401
% hidden layer 1 output = 25 x 1
% hidden layer 2 output = 10 x 1

input_layer = [ ones(m, 1) X ];  % m x 401

hidden_layer = sigmoid( input_layer * Theta1' );  % m x 25

hidden_layer = [ ones(m, 1) hidden_layer ];  % m x 26

output_layer = sigmoid( hidden_layer * Theta2' );  % m x 10

[probability index] = max(output_layer, [], 2);

% index(index == 10) = 0;

p = index;


% =========================================================================


end
