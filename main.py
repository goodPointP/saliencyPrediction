import sys
sys.path.insert(1, '/imageSegmentation')

# package imports

# saliency prediction imports

# image segmentation imports
import imageSegmenation

imageList = ['testPictures/dog.jpg', 'testPictures/2people1.jpg', 'testPictures/car.jpg']

for imagePath in imageList:
    imageName = imagePath.split('/')[-1].split('.jpg')[0]

    heatmap, imagePath = predictEyeGaze(imagePath, sigma=0.9)
    
    masks, segvalues, output = imageSegmenation.segmentTheImage(imagePath)

    heatmapPixelsArray = imageSegmenation.createPixelArrayFromHeatmap(heatmap)

    relevantMaskIndexes = []
    for maskIndex, mask in enumerate(masks):
        if (maskIndex not in relevantMasks):
            for pixel in heatmapPixelsArray:
                if (maskIndex not in relevantMasks):
                    singleObjectMask = imageSegmenation.checkExistanceOfPixelInMask(pixel, mask)
                    if (singleObjectMask):
                        relevantMasks.append(maskIndex)
    
    relevantMasks = imageSegmenation.getRelevantMasks(masks, relevantMaskIndexes)

    foregroundList = []
    for relevantMaskIndex, relevantMask in enumerate(relevantMasks):
        foregroundList.append(imageName, relevantMaskIndex, relevantMask)
        imageSegmenation.createHighQualitySegment(relevantMask)
    
    # create compressed image
    background = imageSegmenation.compressImage(imageName, imagePath)

    imageSegmenation.pasteImages(foregroundList, background, imageName)