# new env "tensorflow gpu" - https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/
# install pixellib - https://pixellib.readthedocs.io/en/latest/

import pixellib 

from pixellib.instance import instance_segmentation

def checkExistanceOfPixelInMask(pixel, mask):
    # first checks if pixel belongs to any of detected objects
    # if TRUE returns the object's mask (2D array of pixels)
    for detectedObject in mask:
        for subObject in detectedObject:
            for maskPixel in subObject:
                if ( all(pixel == maskPixel) ):
                    return subObject #object's mask

segment_image = instance_segmentation()
segment_image.load_model("../../mask_rcnn_coco.h5")
segvalues, output = segment_image.segmentImage("test2People.jpg", output_image_name= "test2Instances3.jpg", mask_points_values = True)
# segvalues, output = segment_image.segmentImage("test2People.jpg")
# segment_image.filter_objects(segvalues, 'person')

onlyMasks = segvalues['masks']