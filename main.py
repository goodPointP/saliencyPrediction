#%%
# package imports

# saliency prediction imports
from inline_inference import *

# image segmentation imports
#import imageSegmenation
from imageSegmentation import *


model = heatmap_inference("models/newnet_model")
# imageList = ['testPictures/dog.jpg', 'testPictures/2people1.jpg', 'testPictures/car.jpg', 'testPictures/cat.jpg',
# 'testPictures/2people2.jpg', 'testPictures/catPerson.jpg', 'testPictures/dog2.jpg', 'testPictures/milanBandic.jpg',
# 'testPictures/videogameScreenshot1.jpg', 'testPictures/videogameScreenshot2.jpg', 'testPictures/woman.jpg']
imageList = ['testPictures/dog.jpg']

for imagePath in imageList:
    imageName = imagePath.split('/')[-1].split('.jpg')[0]
    imWidth, imHeight = getDimensions(imagePath)

    heatmap = model.inline_inference(imagePath, 0.5)
    heatmap = resizeHeatmap(heatmap, (imWidth, imHeight), True, imageName)
    

    masks, segvalues, output = segmentTheImage(imagePath, imageName)

    heatmapPixelsArray = createPixelArrayFromHeatmap(heatmap)
    # convert to set for easier checking
    heatmapPixelsArray = set(heatmapPixelsArray)

    relevantMaskIndexes = []

    for index, mask in enumerate(masks):
        if any([i for i in heatmapPixelsArray if i in mask[0]]):
            relevantMaskIndexes.append(index)


    # for maskIndex, mask in enumerate(masks):
    #     if (maskIndex not in relevantMaskIndexes):
    #         for pixel in heatmapPixelsArray:
    #             if (maskIndex not in relevantMaskIndexes):
    #                 singleObjectMask = checkExistanceOfPixelInMask(pixel, mask)
    #                 if (singleObjectMask):
    #                     relevantMaskIndexes.append(maskIndex)
    
    #relevantMasks = getRelevantMasks(masks, relevantMaskIndexes)

    foregroundList = []
    for relevantMaskIndex in relevantMaskIndexes:
        #foregroundList.append(imageName, relevantMaskIndex, relevantMask)
        createHighQualitySegment(imagePath, imageName, relevantMaskIndex, masks[relevantMaskIndex][0])
    
    # create compressed image
    background = compressImage(imageName, imagePath, debugging=True)

    pasteImages(relevantMaskIndexes, background, imageName)
    print('done with '+imageName)
# %%

# %%
