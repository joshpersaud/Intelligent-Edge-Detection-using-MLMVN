function [ psnr, rmse ] = patchMLMVNFiltLLS(patchSize, smallPatchSize, patchOffset, hidneur_weights, outneur_weights, ImageFileNameInput, ImageFileNameOutput, ImageFileNameIdeal, sec_nums)
% This function performs actual filtering of an image using MLMVN with the
% weights obtained as a result of learning

%sec_nums = 288

% Input parameters:
%
% hidneur_weights - weights for hidden neurons
% outneur_weights - weights for output neurons
% patchSize - 
% size of a squared patch (its actual 2D size is patchSize x patchSize)
% patchOffset - Offset for the following patch compared to the previous one
% (optimal is 1)
% ImageFileNameInput - name of the file (string) with a noisy image
% ImageFileNameOutput - name of the file (string) with an output image
% ImageFileNameIdeal - name of the file with an ideal image (for
% comparison)
%
% Output parameters:
%
% psnr - resulting PSNR after filtering
% rmse - resulting (final) RMSE

% reading of the source image ("noisy")
%crop code for A and C

cropAmount = (patchSize - smallPatchSize) / 2;

%A = imread(ImageFileNameInput);
A =(Crop(cropAmount, ImageFileNameInput));

% size of the source image
[rowa,cola] = size(A); 
% reading of the target image ("clean")
%C = imread(ImageFileNameIdeal); 
C = (Crop(cropAmount, ImageFileNameIdeal));
% size of the source image
[rowb,colb] = size(C); 
% if source and target do not have exactly the same size
if ((rowa~=rowb)||(cola~=colb))
    disp('Sizes of the input and ideal images do not match');
    exit
end

% Preallocation of the resulting image
B = zeros(rowa,cola);

% Preallocation of the matrix contining averaging coefficients (over
% patches)
Av = zeros(rowa,cola);
Outputs2D = zeros(patchSize, patchSize);

h = waitbar(0 );

yStart = 1;

while yStart <= rowa-patchSize
%while yStart <= rowa-smallPatchSize
    xStart = 1;
    
    Progress = yStart/rowa*100;
    waitbar(yStart/rowa,h, sprintf('Progress %6.3f %%', Progress) )
        
    while xStart <= cola-patchSize
    %while xStart <= cola-smallPatchSize 
        
        
        Patch = double(A(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1));
        %Patch = double(A(xStart:xStart+patchSize-1, yStart:yStart+patchSize-1));
        
        Inputs = Patch(:)';
        
        Outputs = MLMVN_test_image(Inputs, hidneur_weights, outneur_weights, sec_nums);
        
        %Outputs2D = reshape (Outputs, [patchSize, patchSize]);
        Outputs2D = reshape (Outputs, [smallPatchSize, smallPatchSize]);
        
        
        % Accumulation of the resulting image data
        %B(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1) = ...
        %    B(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1) + ... 
        %    double(Outputs2D);
        x1 = xStart + double(patchSize/2) - double(smallPatchSize/2);
        x2 = xStart + double(patchSize/2) + double(smallPatchSize/2)-1;
        y1 = yStart + double(patchSize/2) - double(smallPatchSize/2);
        y2 = yStart + double(patchSize/2) + double(smallPatchSize/2)-1;
        
        B(y1:y2, x1:x2) = B(y1:y2, x1:x2) + double(Outputs2D);
        %B(x1:x2, y1:y2) = B(x1:x2, y1:y2) + double(Outputs2D);
        
        % Update of the averaging coefficients
        %Av(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1) = ...
        %    Av(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1) + 1;
        Av(y1:y2, x1:x2) = Av(y1:y2, x1:x2) + 1;
        %Av(x1:x2, y1:y2) = Av(x1:x2, y1:y2) + 1;
        
        % Update of the horizontal starting coordinate
        xStart=xStart+patchOffset;
        
    end
        
    yStart=yStart+patchOffset;
  

end

% Processing of the right-most "strip" of the image from top to bottom
xStart = cola - patchSize + 1;
%xStart = cola - smallPatchSize + 1;
yStart = 1;
    while yStart <= rowa-patchSize+1
    %while yStart <= rowa-smallPatchSize+1
        
        Patch = double(A(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1));
        %Patch = double(A(xStart:xStart+patchSize-1, yStart:yStart+patchSize-1));
        
        Inputs = Patch(:)';
        
        Outputs = MLMVN_test_image(Inputs, hidneur_weights, outneur_weights, sec_nums);
        
        %Outputs2D = reshape (Outputs, [patchSize, patchSize]);
        Outputs2D = reshape (Outputs, [smallPatchSize, smallPatchSize]);
        
        % Accumulation of the resulting image data
        %B(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1) = ...
        %    B(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1) + ... 
        %    double(Outputs2D);
        x1 = xStart + double(patchSize/2) - double(smallPatchSize/2);
        x2 = xStart + double(patchSize/2) + double(smallPatchSize/2)-1;
        y1 = yStart + double(patchSize/2) - double(smallPatchSize/2);
        y2 = yStart + double(patchSize/2) + double(smallPatchSize/2)-1;
        
        B(y1:y2, x1:x2) = B(y1:y2, x1:x2) + double(Outputs2D);
        %B(x1:x2, y1:y2) = B(x1:x2, y1:y2) + double(Outputs2D);
        
        % Update of the averaging coefficients
        %Av(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1) = ...
        %    Av(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1) + 1;
        Av(y1:y2, x1:x2) = Av(y1:y2, x1:x2) + 1;
        %Av(x1:x2, y1:y2) = Av(x1:x2, y1:y2) + 1;
        
        % Update of the vertical starting coordinate
        yStart=yStart+patchOffset;
        
    end

% Processing of the down-most "strip" of the image from left to right, the
% bottom right corner is already processed in the previous section and
% shall not be processed here

yStart = rowa - patchSize + 1;    
%yStart = rowa - smallPatchSize + 1;
xStart = 1;
    while xStart <= rowa-patchSize
    %while xStart <= rowa-smallPatchSize % we do not add 1 here to aviod double processing of the bottom right corner 
        
        Patch = double(A(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1));
        %Patch = double(A(xStart:xStart+patchSize-1, yStart:yStart+patchSize-1));
        
        Inputs = Patch(:)';
        
        Outputs = MLMVN_test_image(Inputs, hidneur_weights, outneur_weights, sec_nums);
        
        %Outputs2D = reshape (Outputs, [patchSize, patchSize]);
        Outputs2D = reshape (Outputs, [smallPatchSize, smallPatchSize]);
        
        
        % Accumulation of the resulting image data
        %B(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1) = ...
        %    B(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1) + ... 
        %    double(Outputs2D);
        x1 = xStart + double(patchSize/2) - double(smallPatchSize/2);
        x2 = xStart + double(patchSize/2) + double(smallPatchSize/2)-1;
        y1 = yStart + double(patchSize/2) - double(smallPatchSize/2);
        y2 = yStart + double(patchSize/2) + double(smallPatchSize/2)-1;
        
        B(y1:y2, x1:x2) = B(y1:y2, x1:x2) + double(Outputs2D);
        %B(x1:x2, y1:y2) = B(x1:x2, y1:y2) + double(Outputs2D);
        
        % Update of the averaging coefficients
        %Av(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1) = ...
        %    Av(yStart:yStart+patchSize-1, xStart:xStart+patchSize-1) + 1;
        Av(y1:y2, x1:x2) = Av(y1:y2, x1:x2) + 1;
        %Av(x1:x2, y1:y2) = Av(x1:x2, y1:y2) + 1;
        
        % Update of the horizontal starting coordinate
        xStart=xStart+patchOffset;
        
    end
    
B = B./Av;

for k1 = 1:rowa
    for k2 = 1:cola
        if(B(k1,k2)>255 && (B(k1,k2)<272))
            B(k1,k2) = 255;
        else
            if B(k1,k2)>271
                B(k1,k2) = 0;
            end
        end
    end
end

%for k1 = 1:rowa
%    for k2 = 1:cola
%        if(B(k1,k2)>255)
%            B(k1,k2) = 0;
%        end
%    end
%end

close(h)

B = uint8(B);

rowa = rowa - cropAmount * 2;
cola = cola - cropAmount * 2;

A = imcrop(A, [(cropAmount + 1) (cropAmount + 1) (rowa - 1) (cola - 1)]);

[rowb,colb] = size(B); 
rowb = rowb - cropAmount * 2;
colb = colb - cropAmount * 2;

B = imcrop(B, [(cropAmount + 1) (cropAmount + 1) (rowb - 1) (colb - 1)]);

[rowc,colc] = size(C); 
rowc = rowc - cropAmount * 2;
colc = colc - cropAmount * 2;

C = imcrop(C, [(cropAmount + 1) (cropAmount + 1) (rowc - 1) (colc - 1)]);


A = imread(ImageFileNameInput);
C = imread(ImageFileNameIdeal); 

figure (1);

imshow(A);

figure (2);

imshow(B);

figure (3);

imshow(C);

imwrite(B, ImageFileNameOutput, 'Compression', 'none');


% RMSE

rmse1 = (double(B(:)) - double(C(:))).^2;
rmse = (sum(rmse1))/(rowa*cola);
rmse = sqrt(rmse);

psnr = 20 * log10(255/rmse);

end
