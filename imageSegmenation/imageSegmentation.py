# new env "tensorflow gpu" - https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/
# install pixellib - https://pixellib.readthedocs.io/en/latest/
#%%
import pixellib 
import numpy as np
from pixellib.instance import instance_segmentation

# CHECK IF PREDICTED GAZE POINT IS CONTAINED IN A DETECTED IMAGE SEGMENT
def checkExistanceOfPixelInMask(pixel, mask):
    # first checks if pixel belongs to any of detected objects
    # if TRUE returns the object's mask (2D array of pixels)
    for detectedObject in mask:
        for subObject in detectedObject:
            for maskPixel in subObject:
                if ( all(pixel == maskPixel) ):
                    return subObject #object's mask


# SEGMENT THE IMAGE
segment_image = instance_segmentation()
segment_image.load_model("../../mask_rcnn_coco.h5")
segvalues, output = segment_image.segmentImage("test2People.jpg", output_image_name= "test2Instances6.jpg", mask_points_values = True)

# gets the detected objects' masks. length of onlyMasks is the number of detected objects
onlyMasks = segvalues['masks'][0]

#%%
# CREATE BOOLEAN MASK FOR - NOT NEEDED?
def createBooleanMaskFromPixelArray(maskPixelArray, imageX, imageY):
    booleanMask = np.zeros((imageY, imageX), np.bool8)
    for indexX, row in enumerate(booleanMask):
        for indexY, pixel in enumerate(row):
            if ([indexY, indexX] in maskPixelArray):
                booleanMask[indexX, indexY] = True
    return booleanMask

# GET A SINGLE MASK
originalMask = onlyMasks[0].tolist()
booleanMask = createBooleanMaskFromPixelArray(originalMask, 1800, 1200)

# %%
# CREATE A HIGH-QUALITY CUT-OUT OF THE SELECTED MASK
import numpy
from PIL import Image, ImageDraw

tupleList = []
for pixel in originalMask:
    tupleList.append((pixel[0], pixel[1]))

# read image as RGB and add alpha (transparency)
im = Image.open("test2PeopleCrop.jpg").convert("RGBA")

# convert to numpy (for convenience)
imArray = numpy.asarray(im)

# create mask
#polygon = [(444,203),(623,243),(691,177),(581,26),(482,42)]
polygon = tupleList
maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
mask = numpy.array(maskIm)

# assemble new image (uint8: 0-255)
newImArray = numpy.empty(imArray.shape,dtype='uint8')

# colors (three first columns, RGB)
newImArray[:,:,:3] = imArray[:,:,:3]

# transparency (4th column)
newImArray[:,:,3] = mask*255

# back to Image from numpy
newIm = Image.fromarray(newImArray, "RGBA")
newIm.save("out.png")


# %%
# COMPRESS THE IMAGE
def compressImage(file, verbose = False):
    
      # Get the path of the file
    filepath = file
      
    # open the image
    picture = Image.open(filepath)
      
    # Save the picture with desired quality
    # To change the quality of image,
    # set the quality variable at
    # your desired level, The more 
    # the value of quality variable 
    # and lesser the compression
    picture.save("Compressed_"+file, 
                 "JPEG", 
                 optimize = True, 
                 quality = 10)
    return

compressImage('test2People.jpg')

# %%
# PASTE (COMBINE) THE LOW-QUALITY BACKGROUND WITH HIGH-QUALITY FOREGROUND(S)
from PIL import Image, ImageDraw, ImageFilter

im1 = Image.open('Compressed_test2People.jpg')
im2 = Image.open('out.png')

im1.paste(im2, (0,0), im2)
im1.save('paste.jpg')

# %%
