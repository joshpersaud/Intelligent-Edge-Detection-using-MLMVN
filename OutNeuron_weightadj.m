
%Weights adjustment for output neuron

function [w_adj, z_c] = OutNeuron_weightadj(X, outneur_weights, phase_d, z_d, y_d, sec_size, N, outneur_num, local_thresh)


%X = matrix of output neuron inputs (N x n), where N=number of learning
%samples, n = number of hidden neurons

%outneur_weights = (n+1 x outneur_num) matrix of output neuron weights

%phase_d = (N x outneur_num) matrix of desired network outputs for all N learning
%samples, expressed as phase

%sec_size = (1 x outneur_num) vector containing the angular sector size of
%each output neuron

%------
%n = size(X, 2);
%N = size(X, 1);

%append a column of 1s to X from the left, making it a (N x n+1) matrix
col_app(1:N) = 1;
col_app = col_app.';
X = [col_app X];

%Compute the weighted sums of the output neuron for all N samples
z_c = X * outneur_weights;

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

        %    ang_err = 2*pi - ang_err;
        %end

        if (ang_err < local_thresh)
            net_err(ii, pp) = 0;
        end
    end
end


%Compute the full SVD of X
%[U,S,V] = svd(X);

%M = n+1
%M = length(w);

%Retain only the first M columns of U, and first M rows of S
%U_hat = U(:, 1:M);
%S_hat = S(1:M, :);

%Construct the pseudo-inverse of S
%S_hpinv = diag(1 ./ diag(S_hat));

%Construct the pseudo-inverse of X
%X_pinv = V * S_hpinv * U_hat';

X_pinv = pinv(X);

w_adj = outneur_weights;
%Go over each output neuron
for pp = 1 : outneur_num

    %LLS: apply X_pinv to delta
    adj_vec = X_pinv * (1 .* net_err(:, pp));

    %the new weights are given by 
    w_adj(:, pp) = outneur_weights(:, pp) + adj_vec;
end

%Construct a vector of current weighted sums for N samples
z_c = X * w_adj;





