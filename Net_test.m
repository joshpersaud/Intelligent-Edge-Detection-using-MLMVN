
function classif_rate = Net_test(X, y_d, hidneur_weights, outneur_weights, win_ang)

%Convert input values into complex numbers on the unit circle
X = exp(1i .* X);

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
win_dist = zeros(N, outneur_num);
for ii=1:N
    
    for pp = 1 : outneur_num
       
        win_dist(ii, pp) = abs(current_phase(ii, pp) - win_ang);

        if (win_dist(ii, pp) > pi)

            win_dist(ii, pp) = 2*pi - win_dist(ii, pp);
        end

    end
end

[min_dist, current_labels] = min(win_dist, [], 2);
current_labels = current_labels - 1;

sovpad = 0;

for ii=1:N
    if (current_labels(ii) == y_d(ii))
        sovpad = sovpad + 1;
    end
end

classif_rate = sovpad / N;
