function trainedNetData = trainAndTest( ...
    net, ...
    trainImgs, validationImgs, testImgs, ...
    maxEpochs, minibatchSize, initialLearnRate, ...
    netName ...
)
%TRAINANDTEST Train and test the given net on the given dataset splits.

%% Get the unique class labels
classes = categories(trainImgs.Labels);

%% Train the model
[trainedNet, trainedNetInfo, resizeTestImgs, options] = trainnet_wrapper( ...
    net, ...
    trainImgs, validationImgs, testImgs, ...
    maxEpochs,  minibatchSize, initialLearnRate ...
);


%% Test the model
accuracy = testnet(trainedNet,resizeTestImgs, "accuracy");

testscores = minibatchpredict(trainedNet, resizeTestImgs);
testpreds = scores2label(testscores, classes);
truetest = testImgs.Labels;

netName = netName.replace("_", "\_");

confusionchart( ...
    truetest, testpreds, ...
    Title = netName + " - Confusion chart" ...
);

%% Show incorrectly classified images
idx = find(testpreds ~= truetest);
if ~isempty(idx)
    numWrongImgs = length(idx);
    for i = 1:numWrongImgs
        figure;
        imshow(readimage(testImgs,idx(i)));
        title({ ...
            netName, ...
            "Class = " + string( truetest(idx(i)) ), ...
            "Prediction = " + string( testpreds(idx(i)) ) ...
        });
    end
end

%% Return a struct with all the relevant stuff
trainedNetData.net = trainedNet;
trainedNetData.info = trainedNetInfo;
trainedNetData.trainingOptions = options;
trainedNetData.testSetAccuracy = accuracy;

end