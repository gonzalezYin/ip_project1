BGR = imread('tif/image3/BGR.tif');
HSV = imread('tif/image3/HSV.tif');
YCrCb = imread('tif/image3/YCrCb.tif');
LAB = imread('tif/image3/LAB.tif');

S_BGR = decorrstretch(BGR,'tol',0.01);
figure;
imshow(S_BGR);
title('BGR with D');
imwrite(S_BGR, 'tif/image3/BGRD.tif');

S_HSV = decorrstretch(HSV,'tol',0.01);
figure;
imshow(S_HSV);
title('HSV with D');
imwrite(S_HSV, 'tif/image3/HSVD.tif');

S_YCrCb = decorrstretch(YCrCb,'tol',0.01);
figure;
imshow(S_YCrCb);
title('YCrCb with D');
imwrite(S_YCrCb, 'tif/image3/YCrCbD.tif');

S_LAB = decorrstretch(LAB,'tol',0.01);
figure;
imshow(S_LAB);
title('LAB with D');
imwrite(S_LAB, 'tif/image3/LABD.tif');