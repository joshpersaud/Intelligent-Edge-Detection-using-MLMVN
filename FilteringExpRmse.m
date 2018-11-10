function [hidneur_weights, outneur_weights, iterations, RMSE] = FilteringExpRmse(X_learn, Y_learn, RMSE_thresh) %x lean is imput y learn is output
% This function utilizes a learning process for iamge filtering based on
% processing of overlapping patches using batch MLMVN-LLS algorithm
% strating from random weights
%
% Input parameters:
% X_learn - Matrix of inputs (input samples are organized row by row)
% Y_learn - Matrix of desired outputs (outputs are organized row by row)
%
% Output parameters :
%
% hidneur_weights - weights for hidden neurons
% outneur_weights - weights for output neurons
% iterations - final number of learning iterations
% RMSE - resulting (final) RMSE

%Learning function parameters
% number of hidden neurons
hidneur_num = 2048;
% number of output neurons
%outneur_num = 225; %use otput size
[rowy, coly] = size(Y_learn);
outneur_num = coly;

% number of sectors
sec_nums = zeros(1,outneur_num);
sec_nums(1,:) = 288;
% threshold for RMSE (global)
% RMSE_thresh = 0.0327;
% RMSE_thresh = 6.5; %can lower to 6/6.5
% threshold for RMSE (local)
local_thresh = 0.0;

    tic
    [hidneur_weights, outneur_weights, iterations, RMSE] = Net_learn_rmse(X_learn, Y_learn, hidneur_num, outneur_num, sec_nums, RMSE_thresh, local_thresh);
    toc
