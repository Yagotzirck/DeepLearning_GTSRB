function custom_tsr_net = create_custom_tsr_net(num_classes)
%CREATE_CUSTOM_TSR_NET Creates and returns a custom model for the GTSR dataset.
custom_tsr_net = dlnetwork;

tempNet = [
    imageInputLayer([128 128 3],"Name","imageinput","Normalization","zscore")

    groupedConvolution2dLayer([5 5],32,"channel-wise","Name","groupedconv","Padding","same","Stride",[2 2],"WeightsInitializer","he")
    convolution2dLayer([1 1],32,"Name","conv_1x1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","batchnorm_0")
    reluLayer("Name","relu_0_0")
    convolution2dLayer([3 3],32,"Name","conv_0_1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_0_1")
    maxPooling2dLayer([2 2],"Name","maxpool","Padding","same","Stride",[2 2])

    convolution2dLayer([3 3],64,"Name","conv_1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])

    convolution2dLayer([3 3],128,"Name","conv_2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])

    convolution2dLayer([3 3],256,"Name","conv_3","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_3")
    maxPooling2dLayer([2 2],"Name","maxpool_3","Padding","same","Stride",[2 2])

    convolution2dLayer([3 3],512,"Name","conv_4","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_4")
    maxPooling2dLayer([2 2],"Name","maxpool_4","Padding","same","Stride",[2 2])


    fullyConnectedLayer(2048,"Name","fc_0","WeightsInitializer","he")
    reluLayer("Name","relu_5")
    dropoutLayer(0.5,"Name","dropout_0")

    fullyConnectedLayer(2048,"Name","fc_1","WeightsInitializer","he")
    reluLayer("Name","relu_6")
    dropoutLayer(0.5,"Name","dropout_1")
    
    fullyConnectedLayer(num_classes,"Name","fc_2","WeightsInitializer","he")
    softmaxLayer("Name","softmax")];

custom_tsr_net = addLayers(custom_tsr_net,tempNet);

custom_tsr_net = initialize(custom_tsr_net);
end