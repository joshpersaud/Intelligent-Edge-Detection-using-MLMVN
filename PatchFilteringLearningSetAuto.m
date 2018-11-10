function [ X_learn, Y_learn] = PatchFilteringLearningSetAuto(patchSize, smallPatchSize, numberOfImages)
% This function is used to add Gaussian noise to images and then to create learning samples
% patchSize - the patch (block) size (if n x n, then n)
% numberof images - the number of images where from patches are taken
% the function randomly generates starting coordinates 
% and grabs n x n noisy patch as inputs and n x n clean patch as a desired output

%q = 4000;
%numberOfSamples = floor(q/numberOfImages); % number of samples from 1 image

numberOfSamples = 100;
q = numberOfSamples * numberOfImages;
% length of the patch conctenated in row
linPatchSize = patchSize * patchSize;
smallLinPatchSize = smallPatchSize * smallPatchSize;
% preallocation of the resulting array 
%LearningSet = zeros(6000, linPatchSize*2);
LearningSet = zeros(q, smallLinPatchSize + linPatchSize);
counter = 0;
cropAmount = (patchSize - smallPatchSize) / 2;

for k = 1:numberOfImages

    %imFile = strcat(' (',num2str(k),')','.tif');
    if (k <10)
        imFile = ['00' num2str(k)];
    elseif k<100
        imFile = ['0' num2str(k)];
    else
        imFile = num2str(k);
    end
    imFile = strcat(imFile,'.tif');
    %imFile = '002.tif';
    


     if (k <10)
        imFile2 = ['100' num2str(k)];
    elseif k<100
        imFile2 = ['10' num2str(k)];
     else
        imFile2 = ['1' num2str(k)];
    end
    imFile2 = strcat(imFile2,'.tif');
    
    A = Crop(cropAmount, imFile);
    B = Crop(cropAmount, imFile2);
   
  
   % Amean = mean(A(:));
   % Asigma = std(A(:));
  %  Nsigma = 0.3 * Asigma;
    [rowa,cola] = size(A);
   % Noise = randn(rowa,cola);
   % Noise = Noise.*Nsigma + Amean;
    %Anoisy = uint8(double(A)+Noise-Amean);
    
%     write noisy images over orginals
%        if (k <10)
%          
%          imwrite(Anoisy, ['00',num2str(k),'.tif']);
%          count = count + 1;
%         
%      elseif k<100
%        
%          imwrite(Anoisy, ['0',num2str(k),'.tif']);
%      else
%      
%          imwrite(Anoisy, [num2str(k),'.tif']);
%        end
    

    % Limits for starting new patches
    rowlim = rowa-patchSize;
    collim = cola-patchSize;

    % generation of the starting coordinates for the learning patches
    rng('shuffle') % seeds random numbers generator based on the current time

    % horizontal starting coordinates
    startx = rand(1,numberOfSamples);
    % transformation to the range 1...rowlim
    startx = floor(startx.*rowlim)+1;

    % vertical starting coordinates
    starty = rand(1,numberOfSamples);
    % transformation to the range 1...collim
    starty = floor(starty.*collim)+1;
 


    % generation of numberOfSamples learning samples
    for k1 = 1:numberOfSamples
        
        counter = counter + 1; % inc counter of samples
        x = startx(k1);
        y = starty(k1);
        % sample1 - patch from the source
        sample1 = A(x:(x + patchSize - 1), y:(y + patchSize - 1));
        sample1 = sample1(:)';
        % sample2 - patch from the target
        x1 = x + double(patchSize/2) - double(smallPatchSize/2);
        x2 = x + double(patchSize/2) + double(smallPatchSize/2)-1;
        y1 = y + double(patchSize/2) - double(smallPatchSize/2);
        y2 = y + double(patchSize/2) + double(smallPatchSize/2)-1;
        %sample2 = A(x:x+patchSize-1, y:y+patchSize-1); original code
        sample2 = B(x1:x2, y1:y2); %new code
        sample2= sample2(:)';
        
        LearningSet(counter, 1:linPatchSize) = sample1;
        %LearningSet(counter, linPatchSize + 1:end) = sample2; original code
       LearningSet(counter, linPatchSize + 1:end) = sample2; %new code
        %288
        %patchmlmvnfitlls
    end
  
end

X_learn = LearningSet(1:q, 1:linPatchSize);
Y_learn = LearningSet(1:q, linPatchSize + 1:linPatchSize + smallLinPatchSize);


end