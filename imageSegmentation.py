# new env "tensorflow gpu" - https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/
# install pixellib - https://pixellib.readthedocs.io/en/latest/
#%%
import pixellib 
import numpy as np
from pixellib.instance import instance_segmentation
from PIL import Image, ImageDraw, ImageFilter

outputsFolder = 'imageSegmenationOut/'

# SEGMENT THE IMAGE
def segmentTheImage(imageName):
    segment_image = instance_segmentation()
    segment_image.load_model("../mask_rcnn_coco.h5")
    segvalues, output = segment_image.segmentImage(imageName, output_image_name= outputsFolder + imageName+"-segmented.jpg", mask_points_values = True)
    masks = segvalues['masks'][0] # gets the detected objects' masks. length of onlyMasks is the number of detected objects
    return (masks, segvalues, output)

def createPixelArrayFromHeatmap(heatmap):
    ## TO DO
    #
    #
    #
    #
    #
    pixelArray = []
    return pixelArray

# CHECK IF PREDICTED GAZE POINT IS CONTAINED IN A DETECTED IMAGE SEGMENT
def checkExistanceOfPixelInMask(pixel, mask):
    # first checks if pixel belongs to any of detected objects
    # if TRUE returns the object's mask (2D array of pixels)
    for detectedObject in mask:
        for subObject in detectedObject:
            for maskPixel in subObject:
                if ( all(pixel == maskPixel) ):
                    return subObject #object's mask

# CREATE BOOLEAN MASK FOR - NOT NEEDED?
def createBooleanMaskFromPixelArray(maskPixelArray, imageX, imageY):
    booleanMask = np.zeros((imageY, imageX), np.bool8)
    for indexX, row in enumerate(booleanMask):
        for indexY, pixel in enumerate(row):
            if ([indexY, indexX] in maskPixelArray):
                booleanMask[indexX, indexY] = True
    return booleanMask

# RETURNS SELECTED RELEVANT MASKS
def getRelevantMasks(allMasks, relevantMaskIndexes):
    relevantMasks = []
    for maskIndex, mask in enumerate(allMasks):
        if (maskIndex in relevantMaskIndexes):
            relevantMasks.append(mask.tolist())
    
    return relevantMasks
    # originalMask = allMasks[maskIndex].tolist()
    #booleanMask = createBooleanMaskFromPixelArray(originalMask, 1800, 1200)

# %%
# CREATE A HIGH-QUALITY CUT-OUT OF THE SELECTED MASK
def createHighQualitySegment(imageName, relevantMaskIndex, originalMask):
    tupleList = []
    for pixel in originalMask:
        tupleList.append((pixel[0], pixel[1]))

    # read image as RGB and add alpha (transparency)
    im = Image.open(imageName).convert("RGBA")

    # convert to numpy (for convenience)
    imArray = np.asarray(im)

    # create mask
    #polygon = [(444,203),(623,243),(691,177),(581,26),(482,42)]
    polygon = tupleList
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = np.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = np.empty(imArray.shape,dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]

    # transparency (4th column)
    newImArray[:,:,3] = mask*255

    # back to Image from numpy
    newIm = Image.fromarray(newImArray, "RGBA")
    newIm.save(outputsFolder+imageName+"-cutoutNumber"+relevantMaskIndex+".png")
    return


# %%
# COMPRESS THE IMAGE
def compressImage(imageName, filepath, verbose = False):
      
    # open the image
    picture = Image.open(filepath)
      
    # Save the picture with desired quality
    # To change the quality of image,
    # set the quality variable at
    # your desired level, The more 
    # the value of quality variable 
    # and lesser the compression
    outputFilename = outputsFolder + imageName+"-compressed.jpg"
    picture.save(outputFilename, 
                 "JPEG", 
                 optimize = True, 
                 quality = 10)
    return outputFilename

# %%
# PASTE (COMBINE) THE LOW-QUALITY BACKGROUND WITH HIGH-QUALITY FOREGROUND(S)
def pasteImages(foregrounds, background, imageName):
    im1 = Image.open(outputsFolder+'Compressed_test2People.jpg')

    for foreground in foregrounds:
        im2 = Image.open(outputsFolder+imageName+"-cutoutNumber"+foreground)
        im1.paste(im2, (0,0), im2)
        im1.save(outputsFolder+"final-"+imageName+".jpg")