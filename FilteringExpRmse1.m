
function [hidneur_weights, outneur_weights, iterations, RMSE] = FilteringExpRmse1(X_learn, Y_learn, hidneur_weights, outneur_weights, RMSE_thresh)
% This function utilizes a learning process for iamge filtering based on
% processing of overlapping patches using batch MLMVN-LLS algorithm
% strarting from already existing weights
%
% Input parameters:
%
% X_learn - Matrix of inputs (input samples are organized row by row)
% Y_learn - Matrix of desired outputs (outputs are organized row by row)
% hidneur_weights - weights for hidden neurons
% outneur_weights - weights for output neurons
%
% Output parameters:
%
% hidneur_weights - weights for hidden neurons
% outneur_weights - weights for output neurons
% iterations - final number of learning iterations
% RMSE - resulting (final) RMSE

%Learning function parameters
% number of hidden neurons
hidneur_num = 2048;
% number of output neurons
% outneur_num = 225;
[rowy, coly] = size(Y_learn);
outneur_num = coly;
% number of sectors
sec_nums = zeros(1,outneur_num);
sec_nums(1,:) = 288;
% threshold for RMSE (global)
% RMSE_thresh = 0.0327;
%RMSE_thresh = 7.1;
%RMSE_thresh = 6.18;
% threshold for RMSE (local)
local_thresh = 0.0;

    tic
    [hidneur_weights, outneur_weights, iterations, RMSE] = Net_learn_rmse1(X_learn, Y_learn, hidneur_weights, outneur_weights, sec_nums, RMSE_thresh, local_thresh);
    toc
    
