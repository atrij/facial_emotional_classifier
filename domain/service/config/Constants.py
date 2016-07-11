class Constants:

    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness",
                "surprise"]

    # Dataset Paths
    datasetPathEmotions = "source_emotions"
    datasetPathImages = "source_images"

    # PreProcessing Methods
    grayScaleConversion = "GRAY_SCALE_CONVERSION"
    faceDetectionHAAR = "FACE_DETECTION_HAAR"
    averaging = "AVERAGING"
    gaussianBlur = "GAUSSIAN_BLUR"
    medianBlur = "MEDIAN_BLUR" # Use for salt-pepper noise
    bilateralFiltering = "BILATERAL_FILTERING" # highly effective in noise reduction while keeping edges sharp. It is very slow
    histogramEqualisation = "HISTOGRAM_EQUALISATION"

    #Datasets
    cohn_Kanade_extended = "CK+"

    # Descriptor Extraction Methods
    sift = "SIFT"

    #Dimensionality Reduction Methods
    pca = "PCA"

    # Descriptor Pooling Methods
    bagOfVisualWords = "BoVW"

    # Lists of methods/databases
    datasetList = [cohn_Kanade_extended]
    desriptorExtractorMethodList = [sift]
    dimensionalityReductionMethodList = [pca]
    descriptorPoolingMethodList = [bagOfVisualWords]