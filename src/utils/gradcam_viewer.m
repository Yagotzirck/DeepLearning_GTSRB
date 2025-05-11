function gradcam_viewer(net, classes, image_path)
%GRADCAM_VIEWER ref. https://it.mathworks.com/help/deeplearning/ug/gradcam-explains-why.html
%   Call example:
%   gradcam_viewer(transferNetData1.net, classes, "dataset/images/02/00002_00001_00028.png");
inputSize = net.Layers(1).InputSize(1:2);
img = imread(image_path);
img = imresize(img, inputSize);

scores = predict(net, single(img));
Y = scores2label(scores, classes);

channel = find(Y == categorical(classes));
map = gradCAM(net, img, channel);

imshow(img);
hold on;
imagesc(map, 'AlphaData', 0.5);
colormap jet
hold off;
title("Grad-CAM");
end