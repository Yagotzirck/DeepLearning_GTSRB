function crop_traffic_signs( ...
    datasetDir, imgs_dirName, cropped_dirName, adjusted_CSV_name ...
)
%CROP_TRAFFIC_SIGNS Helper function called by create_datastore().
%   CROP_TRAFFIC_SIGNS Use the enclosing boxes defined in the csv files
%   included with the traffic signs' dataset to crop the images.

%% Create the directories which will contain the cropped images
mkdir(datasetDir, cropped_dirName);

% Classes are named/numbered in the range 0-42, and there's one folder
% per class in the imgs_dirName subfolder
cropped_dirPath = fullfile(datasetDir, cropped_dirName);
for i = 0:42
    mkdir(cropped_dirPath, string(i));
end

%% Crop the images
imgsInfo = readtable( fullfile(datasetDir, adjusted_CSV_name) );

srcFiles = string(imgsInfo.Path);
destFiles = replace(srcFiles, imgs_dirName, cropped_dirName);

numImgs = height(imgsInfo);

% Create crop rects: each col. is a rectangle, for a total of numImgs rows.
rects = [
    imgsInfo.Roi_X1';                       % xmin
    imgsInfo.Roi_Y1';                       % ymin
    imgsInfo.Roi_X2' - imgsInfo.Roi_X1';    % width
    imgsInfo.Roi_Y2' - imgsInfo.Roi_Y1'     % height
];

parfor i=1:numImgs
    imgSrc = imread(srcFiles(i));
    imgCropped = imcrop(imgSrc, rects(:,i));
    imwrite(imgCropped, destFiles(i));
end

end