function [ Outputs ] = MLMVN_test_image( X, hidneur_weights, outneur_weights, sec_nums )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here



%Determine sector size (angle), separately for each output neuron.
%sec_size is a (1 x outneur_num) vector
sec_size = 2*pi / sec_nums;

secsize1 = sec_size;

%Convert input values into complex numbers on the unit circle
X = exp(1i .* X * secsize1);

%Determine the number of testing samples
N = size(X, 1);

%Determine the number of output neurons
outneur_num = size(outneur_weights, 2);

%append a column of 1s to X from the left
%app_X
col_app(1:N) = 1;
col_app = col_app.';
app_X = [col_app X];

%Compute the output of hidden neurons for all samples
hid_outmat = app_X * hidneur_weights;

%Move outputs to the unit circle
hid_outmat = hid_outmat ./ abs(hid_outmat);

%append a column of 1s to hid_outmat
hid_outmat = [col_app hid_outmat];

%Compute the output of the network
z_outneur = hid_outmat * outneur_weights;

%We will now apply the "winner take it all" principle in the following
%manner: the output neuron whose output is closest to pi/2 determines the
%output class
%win_ang = pi/2;
current_phase = angle(z_outneur);

    current_phase = mod(current_phase, 2*pi);

    Outputs = zeros(N, outneur_num);
    for pp = 1 : outneur_num
        
        Outputs(:, pp) = floor(current_phase(:, pp) ./ sec_size);
    end

        
%    Outputs = floor(current_phase ./ sec_size);


end

