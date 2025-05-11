function imds = create_datastore(crop_imgs)
%CREATE_DATASTORE Creates an ImageDatastore object from "archive.zip".
%
% CREATE_DATASTORE(crop_imgs) unzips (in the folder specified by datasetDir)
% the archive "archive.zip" retrieved from this page:
% https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
% Then, it creates an ImageDatastore object based on its contents.
%
% The distinction between "Train" and "Test" folders is disregarded and
% they're merged as a whole dataset, leaving the freedom to decide how
% to partition it into training/validation/test sets afterwards.
%
% The only input parameter is the boolean value crop_imgs; if it's
% true, the images are cropped according to the bounding boxes defined in
% the .csv files included with the dataset.

%% files and folders name definitions
dataset_archiveName = "archive.zip";
datasetDir = "dataset";
imgs_dirName = "images";
meta_dirName = "Meta";
cropped_dirName = "cropped";
adjusted_CSV_name = "dataset.csv";


%% Extract and adjust the dataset if it hasn't been done already
if ~exist(datasetDir, "dir")
    fprintf( ...
        "Extracting %s in folder '/%s'... ", ...
        dataset_archiveName, datasetDir ...
    );
    unzip(dataset_archiveName, datasetDir);
    disp("done");

    fprintf("Adjusting dataset... ");
    adjust_dataset(datasetDir, imgs_dirName, adjusted_CSV_name);
    disp("done");

    %% Crop images
    fprintf('Cropping images... ');
    crop_traffic_signs( ...
        datasetDir, imgs_dirName, cropped_dirName, adjusted_CSV_name ...
    );
    disp("done");

    %% Rename class folders having only one digit by adding a leading "0"
    %  in order to keep lexicographical order in the returned imds object
    dataset_dirs = [imgs_dirName, cropped_dirName];
    for i = 1:2
        currImgsDir = dataset_dirs(i);
        for j = 0:9
            movefile( ...
                fullfile(datasetDir, currImgsDir, string(j)), ...
                fullfile(datasetDir, currImgsDir, sprintf("%02d", j)) ...
            );
        end
    end


    %% Rename Meta images such that all images' names have two digits.
    %  For instance, "1.png" becomes "01.png".
    for i = 0:9
        movefile( ...
            fullfile(datasetDir, meta_dirName, string(i) + ".png"), ...
            fullfile(datasetDir, meta_dirName, sprintf("%02d", i) + ".png") ...
        );
    end
    
end

%% Use cropped images if crop_imgs = true; use uncropped images otherwise.
if crop_imgs
    pathToImages = fullfile(datasetDir, cropped_dirName);
else
    pathToImages = fullfile(datasetDir, imgs_dirName);
end

%% Create the datastore using subfolder names as class labels
imds = imageDatastore( ...
    pathToImages, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames" ...
);

end