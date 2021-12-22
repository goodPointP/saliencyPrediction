#%%
# package imports

# saliency prediction imports
from inline_inference import *

# image segmentation imports
#import imageSegmenation
from imageSegmentation import *

def getListOfImages():
    with open('../../Datasets/EvaluationDataset/ALLSTIMULI/images.txt') as f:
        lines = f.readlines()
    
    listOfImages = []
    for line in lines:
        listOfImages.append('../../Datasets/EvaluationDataset/ALLSTIMULI/' + line.split('\n')[0])
    
    return listOfImages

model = heatmap_inference("models/newnet_model")
# imageList = ['testPictures/dog.jpg', 'testPictures/2people1.jpg', 'testPictures/car.jpg', 'testPictures/cat.jpg',
# 'testPictures/2people2.jpg', 'testPictures/catPerson.jpg', 'testPictures/dog2.jpg', 'testPictures/milanBandic.jpg',
# 'testPictures/videogameScreenshot1.jpg', 'testPictures/videogameScreenshot2.jpg', 'testPictures/woman.jpg']
# imageList = ['testPictures/istatic_coast_landscape_outdoor_dsc03095.jpg']
imageList = getListOfImages()
succesfullyProcessedList = []
imagesWithNoMasks = []

for imagePath in imageList:
    try:
        imageName = imagePath.split('/')[-1].split('.jpg')[0]
        imWidth, imHeight = getDimensions(imagePath)

        heatmap = model.inline_inference(imagePath, 0.8)
        heatmap = resizeHeatmap(heatmap, (imWidth, imHeight), True, imageName)
        

        masks, segvalues, output = segmentTheImage(imagePath, imageName)

        heatmapPixelsArray = createPixelArrayFromHeatmap(heatmap)
        # convert to set for easier checking
        heatmapPixelsArray = set(heatmapPixelsArray)

        relevantMaskIndexes = []

        for index, mask in enumerate(masks):
            if any([i for i in heatmapPixelsArray if i in mask[0]]):
                relevantMaskIndexes.append(index)
        
        if len(relevantMaskIndexes)>0:
            foregroundList = []
            for relevantMaskIndex in relevantMaskIndexes:
                #foregroundList.append(imageName, relevantMaskIndex, relevantMask)
                createHighQualitySegment(imagePath, imageName, relevantMaskIndex, masks[relevantMaskIndex][0])
            
            # create compressed image
            background = compressImage(imageName, imagePath, debugging=False)

            pasteImages(relevantMaskIndexes, background, imageName)
            print('done with '+imageName)
            succesfullyProcessedList.append(imageName)
        else:
            imagesWithNoMasks.append(imageName)
    except:
        pass
    

textfile = open("successfullyProcessedImages.txt", "w")

for element in succesfullyProcessedList:

    textfile.write(element + "\n")

textfile.close()


textfile = open("unsuccessfullyProcessedImages.txt", "w")

for element in imagesWithNoMasks:

    textfile.write(element + "\n")

textfile.close()
