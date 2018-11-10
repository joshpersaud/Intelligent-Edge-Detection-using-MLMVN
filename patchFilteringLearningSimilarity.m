function [ LearningSet1 ] = patchFilteringLearningSimilarity( LearningSet, threshold1, threshold2 )
%patchFilteringLearningSimilarity is looking for similar patches (samples)
%in LearningSet. Similarity is determined through SSD - sum of squared
%differences. If noormalized SSD does not exceed threshold, then patches
%are considered similar and only one representative of similar patches then
%remains in the resulting LearningSet1.
%  

% size of the learning Set
[numberOfSamples, patchSize] = size(LearningSet);
patchSize = patchSize/2;

% Extraction of all noisy patches
Patches = LearningSet(:,1:patchSize);
stdPatches = std(Patches, 1, 2);
meanPatches = mean(Patches, 2);
PatchFlags = ones(numberOfSamples,1);
PatchFlags(1) = 0;
counter = 1;

h = waitbar(0 );

for k =1:numberOfSamples-1
    waitbar(k/numberOfSamples,h, sprintf('Progress %6d ', k) )
    if (PatchFlags(k)==0)
        Patch1 = Patches(k,:);
        counter = counter + 1;
        LearningSet1(counter,:) = LearningSet(k,:);
        for k1 = k+1:numberOfSamples
            
            if (abs(stdPatches(k)-stdPatches(k1)>threshold1) && abs(meanPatches(k)-meanPatches(k1))>threshold2)
                             
                PatchFlags(k1) = 0;
            end
            %nSSD = (sum((Patch1-Patch2)^2))/patchSize;
            %nSSD = sqrt(nSSD);
            %if nSSD > threshold
            %    PatchFlags(k1) = 0;
            %end
          
        end
    end
    
end

close(h);

end

