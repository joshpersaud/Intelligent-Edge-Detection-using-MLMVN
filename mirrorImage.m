function [ C ] = mirrorImage( image,kernel )
% This function extends an image contained in the image matrix
% mirroring it in all directions to be able then to use a local window
% of the kernel x kernel size (kernel = 3, 5, or 7) around each pixel

C = image;
imSize = size(C);
rowSize = imSize(1);
colSize = imSize(2);




%get the right size mirror depending on the size of the window
if kernel == 3
    mirror = 3;
end
if kernel == 5
    mirror = 4;
     
end
if kernel == 7
    mirror = 5;
    
end

%this section mirrors the top and left
diff = 1;% used to help grab the right row for mirroring of different size windows
for l = 3: mirror
C = [ C(1:rowSize,l-diff) C];
colSize = colSize + 1;
C = [ C(l-diff,1:colSize); C];
rowSize = rowSize + 1;

diff = diff - 1;
end

%this section mirrors the bottom and right
diff = 0;% used to grab the correct row because it changes with each
            %increase in column and row
 for t = 1 : mirror - 2
C = [C C(1:rowSize, colSize - t - diff)];
colSize = colSize + 1;
diff = diff + 1;
 end
 diff = 0;
 for s = 1 : mirror - 2
C = [C ; C(rowSize - s - diff, 1:colSize)];
rowSize = rowSize + 1;
diff = diff + 1;
 end

end

