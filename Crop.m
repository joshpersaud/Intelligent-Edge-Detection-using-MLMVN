
function [imFile] = Crop(cropAmount, imFile)

    %Crop and mirror image
    %Inefficient Code 
    %imFile = num2str(k) +'.tif';
    %imFile = 'door.png';
    %InfoImage = imfinfo(imFile);
    %Width = InfoImage.Width;
    %Height = InfoImage.Height;
    %imFile = imread(imFile);
    %imFile = cat(1,(flip(imFile,1)),imFile,flip(imFile,1));
    %imFile = [(flip(imFile,2)),imFile,flip(imFile,2)];
    %imFile = imcrop(imFile,[(Width - 50) (Height - 50) (Width + 50) (Height + 50)]);
    %imshow (imFile);
    
    %efficient code
    %crop amount for patchMLNVN should be patchSize - smallPatchSize 
    %InfoImage = imfinfo(imFile);
    %Width = InfoImage.Width;
    %Height = InfoImage.Height;
    imFile = imread(imFile);
    [Height, Width] = size(imFile);

    %x = 15; crop amount
    imUp = flip(imcrop(imFile, [(0) (0) (Width) (cropAmount)]), 1);
    imDown = flip(imcrop(imFile, [(0) (Height - cropAmount + 1) (Width) (Height)]), 1);
      
    imFile = cat(1, imUp, imFile, imDown);
      
    imRight = flip(imcrop(imFile, [(Width - cropAmount + 1) (1) (Width) (Height + 2 * cropAmount)]), 2);
    imLeft = flip(imcrop(imFile, [(0) (0) (cropAmount) (Height + 2 * cropAmount)]), 2);
     
    imFile = double([imLeft, imFile, imRight]);

end