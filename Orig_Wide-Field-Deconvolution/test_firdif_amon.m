function [r] = test_firdif_amon(y,gamma,smt)
% This function implements the first diffrences method
% Inputs:  y - row vector, the measured fluorescence trace; 
% if y is a matrix each row in the matrix is treated as a fluorescence trace
% y - size Txn
% gamma - number, the calcium decay between two measurment points
% smt - number, smoothing (number of points to use) on the algorith rate result    
% Returns the deconvolved rate r_final (T-1xn)
% the first point of the calcium r1 (1xn)
% and the offset beta 0 (1Xn)


T = size(y,1);
D = [zeros(1,T); [-gamma*eye(T-1) zeros(T-1,1)]] + eye(T);

% deconvolve
r = D*y;

end

