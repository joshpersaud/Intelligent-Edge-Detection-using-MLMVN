
function [hidneur_weights, outneur_weights, iterations] = FilteringExp(X_learn, Y_learn)

%Learning function parameters
hidneur_num = 1024;
outneur_num = 25;
sec_nums = [288 288 288 288 288 288 288 288 288 288 288 288 288 288 288 288 288 288 288 288 288 288 288 288 288];
% RMSE_thresh = 0.0327;
RMSE_thresh = 0.01;
local_thresh = 0.0;

trials_num = 1;

%classif_rates = zeros(trials_num, 1);
iterations = 0;

   
    [hidneur_weights, outneur_weights, iterations] = Net_learn(X_learn, Y_learn, hidneur_num, outneur_num, sec_nums, RMSE_thresh, local_thresh);
    %classif_rates(nn) = Net_test(X_test, Y_test, hidneur_weights, outneur_weights, pi/2);
