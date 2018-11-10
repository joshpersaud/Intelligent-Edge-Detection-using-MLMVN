
%Error back-propagation, from output neuron to hidden layer

function [hid_errmat] = ErrBackProp(hid_outmat, outneur_weights, phase_d, z_d, y_d, sec_size, N, n, outneur_num, local_thresh, abs_hid_outmat)


%hid_outmat = (N x n) matrix of hidden neuron outputs, where
%N = number of learning samples, n = number of hidden neurons

%outneur_weights = (n+1 x outneur_num) matrix of output neuron weights

%z_d = (N x outneur_num) matrix of desired network outputs, expressed as weighted
%sums lying on the unit circle

%sec_size = (1 x outneur_num) vector containing the angular sector size of
%each output neuron

%N = size(hid_outmat, 1);
%n = size(hid_outmat, 2);

%append a column of 1s to hid_outmat from the left, making it a (N x n+1) matrix
col_app(1:N) = 1;
col_app = col_app.';
hid_outmat = [col_app hid_outmat];


%Compute the weighted sums of the output neurons for all N samples
z_c = hid_outmat * outneur_weights;

z_c_mag = abs(z_c);

%Move output to the unit circle
z_c = z_c ./ z_c_mag; % abs(z_c);


%Compute network error for all samples
net_err = z_d - z_c;

current_phase = angle(z_c);
current_phase = mod(current_phase, 2*pi);

%for ii=1:N
%    
%    for pp = 1 : outneur_num
%        
%        if (current_phase(ii, pp) < 0)
%            current_phase(ii, pp) = current_phase(ii, pp) + 2*pi;
%        end
%    end
%end


%Zero the error for those samples whose label matches the desired label
%current_labels = floor(current_phase ./ sec_size);

for ii=1:N
    
    for pp = 1 : outneur_num
        %if (current_labels(ii, pp) == y_d(ii, pp))
        %     net_err(ii) = 0;
        %end

        ang_err = abs(current_phase(ii, pp) - phase_d(ii, pp));
        
        ang_err = mod(ang_err, pi);

        %if (ang_err > pi)
        %
        %    ang_err = 2*pi - ang_err;
        %end

        if (ang_err < local_thresh)
            net_err(ii, pp) = 0;
        end
    end
end


hid_errmat = zeros(N, n);
for pp = 1 : outneur_num
    %Construct a (1 x n) vector containing the reciprocal weights w1 to wn of
    %output neuron pp
    outneur_rWvec(1, 1:n) = ((outneur_weights(2:n+1, pp)).^-1).';
    hid_errmat = hid_errmat + (1 .* (net_err(:, pp) ./ (n+1))) * outneur_rWvec;
end

% Normlization of the hidden neurons errors by the absolute values of 
% their current weighted sums
hid_errmat=hid_errmat./abs_hid_outmat;
    
%Compute the error of each hidden neuron for all samples. The result is a
%(N x n) matrix, where each column represent a single hidden neuron

%---
