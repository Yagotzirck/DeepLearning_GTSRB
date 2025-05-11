function adjust_dataset(datasetDir, imgs_dirName, adjusted_CSV_name)
%ADJUST_DATASET Helper function called by create_datastore().

%% Rename the "Train/" folder to imgs_dirName
movefile( ...
    fullfile(datasetDir, "Train"), ...
    fullfile(datasetDir, imgs_dirName) ...
);

%% Modify the training images' paths to reflect the folder name change
trainImgsInfo = readtable( fullfile(datasetDir, "Train.csv") );
imgPaths = trainImgsInfo.Path;

newTrainImgPaths = fullfile( ...
    datasetDir, ...
    replace(imgPaths, "Train", imgs_dirName) ...
);

trainImgsInfo.Path = newTrainImgPaths;

%% Move each image from the "Test/" folder to the imgs_dirName subfolder
%  corresponding to its class
testImgsInfo = readtable( fullfile(datasetDir, "Test.csv") );
imgClassIds = testImgsInfo.ClassId;
imgPaths = testImgsInfo.Path;

newPaths = fullfile( ...
    datasetDir, ...
    imgs_dirName, ...
    string(imgClassIds), ...
    replace(imgPaths, "Test/", "/Test_") ...
);

imgPaths = fullfile(datasetDir, imgPaths);
numTestImgs = height(testImgsInfo);

for i = 1:numTestImgs
    movefile(imgPaths{i}, newPaths{i});
end

% Modify the test images' paths to reflect the changed path
testImgsInfo.Path = newPaths;

%% Delete the "Test/" folder and .csv files
delete( fullfile(datasetDir, "Train.csv") );
delete( fullfile(datasetDir, "Test.csv") );
delete( fullfile(datasetDir, "Test", "GT-final_test.csv") );

rmdir( fullfile(datasetDir, "Test") );

%% Balance the dataset by reducing the number of images for each class.
%
% We're going to do this by reading the number of images in the least
% populated class (maxImgsPerClass), then we pick the first
% maxImgsPerClass images from each class sorted by their bounding box's
% resolution in descending order; in this way, we also exclude the
% lower resolution images from the dataset.
imgsInfo = [
    trainImgsInfo;
    testImgsInfo
];

imgClassIds = imgsInfo.ClassId;
classIds = unique(imgClassIds);
numClasses = length(classIds);
numImgsPerClass = zeros(numClasses, 1);

maxImgsPerClass = inf;
for i = 1:numClasses
    numImgsPerClass(i) = sum(imgClassIds == classIds(i));
    if numImgsPerClass(i) < maxImgsPerClass
        maxImgsPerClass = numImgsPerClass(i);
    end
end

imgsInfo.BboxSize = ...
        (imgsInfo.Roi_X2 - imgsInfo.Roi_X1) .* ... % width
        (imgsInfo.Roi_Y2 - imgsInfo.Roi_Y1) ...    % height
    ;

imgsInfo_balanced = table();
imgsToDelete = [];

for i = 1:numClasses
    currClassTable = imgsInfo(imgsInfo.ClassId == classIds(i), :);
    currClassTable = sortrows(currClassTable, 'BboxSize', 'descend');

    imgsInfo_balanced = [
        imgsInfo_balanced;
        currClassTable(1:maxImgsPerClass, :)
    ];

    if numImgsPerClass(i) > maxImgsPerClass
        imgsToDelete = [
            imgsToDelete;
            string( ...
                table2array( ...
                    currClassTable(maxImgsPerClass + 1 : end, 'Path') ...
                ) ...
            ) ...
        ];
    end
end

%% Delete images with smaller bounding box
numImgsToDelete = length(imgsToDelete);
for i = 1:numImgsToDelete
    delete(imgsToDelete(i));
end


%% Brighten images
imgPaths = string(imgsInfo_balanced.Path);
numImgs = length(imgPaths);
parfor i = 1:numImgs
    curr_img = imread(imgPaths(i));
    curr_img = imlocalbrighten(curr_img);
    imwrite(curr_img, imgPaths(i));
end


%% Save a new .csv file reflecting the changes
imgsInfo_balanced = removevars(imgsInfo_balanced, 'BboxSize');
writetable( ...
    imgsInfo_balanced, ...
    fullfile(datasetDir, adjusted_CSV_name) ...
);
end