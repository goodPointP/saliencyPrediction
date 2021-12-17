#%%
# package imports

# saliency prediction imports
from inference import *

# image segmentation imports
#import imageSegmenation
from imageSegmentation import *

#%%
imageList = ['testPictures/dog.jpg', 'testPictures/2people1.jpg', 'testPictures/car.jpg']

for imagePath in imageList:
    imageName = imagePath.split('/')[-1].split('.jpg')[0]

    heatmap = predictEyeGaze(imagePath, sigma=0.9) # SEBASTIAN I AM YOURS
    
    masks, segvalues, output = segmentTheImage(imagePath)

    heatmapPixelsArray = createPixelArrayFromHeatmap(heatmap)

    relevantMaskIndexes = []
    for maskIndex, mask in enumerate(masks):
        if (maskIndex not in relevantMasks):
            for pixel in heatmapPixelsArray:
                if (maskIndex not in relevantMasks):
                    singleObjectMask = checkExistanceOfPixelInMask(pixel, mask)
                    if (singleObjectMask):
                        relevantMasks.append(maskIndex)
    
    relevantMasks = getRelevantMasks(masks, relevantMaskIndexes)

    foregroundList = []
    for relevantMaskIndex, relevantMask in enumerate(relevantMasks):
        foregroundList.append(imageName, relevantMaskIndex, relevantMask)
        createHighQualitySegment(relevantMask)
    
    # create compressed image
    background = compressImage(imageName, imagePath)

    pasteImages(foregroundList, background, imageName)
# %%
