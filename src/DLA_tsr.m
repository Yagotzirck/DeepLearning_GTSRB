% Driver script for the whole traffic signs classification project.
clc;
close all;
clear;

%% Flags and settings
flag_use_cropped_imgs = true;

trainSplitPerc =        .6;
validationSplitPerc =   .2;
%testSplitPerc =         .2;

flag_show_classes_distributions = false;

flag_retrain_pretrainedNet1 =   false;
flag_retrain_pretrainedNet2 =   false;
flag_retrain_customNet =        false;

trainedModelsDir = "trained_models";
pretrainedNetName1 = "mobilenetv2";
pretrainedNetName2 = "resnet18";
customNetName = "custom_tsr_net";


%% Create the datastore
imds = create_datastore(flag_use_cropped_imgs);
if flag_show_classes_distributions
    show_classes_distributions(imds);
end


%% Partition the dataset into train/validation/test splits
rng default; % Keep the splits consistent each time the script is launched

[trainImgs,validationImgs,testImgs] = splitEachLabel( ...
    imds, ...
    trainSplitPerc, ...
    validationSplitPerc, ...
    "randomized" ...
);


%% Create the folder where the trained model(s) will be saved
if ~exist(trainedModelsDir, "dir")
    mkdir(trainedModelsDir);
end


%% Get the unique class labels
classes = categories(imds.Labels);
num_classes = numel(classes);


%% Perform transfer learning on the 1st pretrained network (mobilenetv2)
if  ~exist( fullfile(trainedModelsDir, pretrainedNetName1 + ".mat"), "file") || ...
    flag_retrain_pretrainedNet1

    pretrainedNet1 = imagePretrainedNetwork( ...
        pretrainedNetName1, ...
        NumClasses = num_classes ...
    );
    
    transferNetData = trainAndTest( ...
        pretrainedNet1, ...
        trainImgs, validationImgs, testImgs, ...
        10, ...     % maxEpochs
        64, ...     % minibatchSize
        0.001, ...  % initialLearnRate
        pretrainedNetName1 ...
    );
    
    % Save the trained model
    save( ...
        fullfile(trainedModelsDir, pretrainedNetName1), ...
        "transferNetData" ...
    );
else
    load( ...
        fullfile(trainedModelsDir, pretrainedNetName1 + ".mat"), ...
        "transferNetData" ...
    );
end


%% Perform transfer learning on the 2nd pretrained network (Resnet18)
if  ~exist( fullfile(trainedModelsDir, pretrainedNetName2 + ".mat"), "file") || ...
    flag_retrain_pretrainedNet2

    pretrainedNet2 = imagePretrainedNetwork( ...
        pretrainedNetName2, ...
        NumClasses = num_classes ...
    );
    
    transferNetData2 = trainAndTest( ...
        pretrainedNet2, ...
        trainImgs, validationImgs, testImgs, ...
        10, ...     % maxEpochs
        64, ...     % minibatchSize
        0.001, ...  % initialLearnRate
        pretrainedNetName2 ...
    );
    
    % Save the trained model
    save( ...
        fullfile(trainedModelsDir, pretrainedNetName2), ...
        "transferNetData2" ...
    );
else
    load( ...
        fullfile(trainedModelsDir, pretrainedNetName2 + ".mat"), ...
        "transferNetData2" ...
    );
end


%% Train the custom network from scratch
if  ~exist( fullfile(trainedModelsDir, customNetName + ".mat"), "file") || ...
    flag_retrain_customNet

    customNet = create_custom_tsr_net(num_classes);
    
    customNetData = trainAndTest( ...
        customNet, ...
        trainImgs, validationImgs, testImgs, ...
        30,  ...    % maxEpochs
        32, ...     % minibatchSize
        0.001, ...  % initialLearnRate
        customNetName ...
    );
    
    % Save the trained model
    save( ...
        fullfile(trainedModelsDir, customNetName), ...
        "customNetData" ...
    );
else
    load( ...
        fullfile(trainedModelsDir, customNetName + ".mat"), ...
        "customNetData" ...
    );
end



%% Helper functions

function show_classes_distributions(imds)
%Displays a bar char representing the number of images in each class.

classes_distr = countEachLabel(imds);
%minImgsCount = min(classes_distr.Count);

figure;
bar(classes_distr.Count);

set( ...
    gca, ...
    XTickLabels = classes_distr.Label, ...
    XTick = 1:numel(classes_distr.Label) ...
);

title('Number of images for each class');
xlabel('Class labels');
ylabel('Number of images');
xtickangle(90); % Rotate class labels for better readability
end