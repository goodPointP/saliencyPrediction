# install pixellib HERE: https://pixellib.readthedocs.io/en/latest/

import pixellib 

from pixellib.instance import instance_segmentation

segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5")
segment_image.segmentImage("test2People.jpg", output_image_name= "test2Instances2.jpg")
# segvalues, output = segment_image.segmentImage("imageSegmenation/testImage.jpeg")
# segment_image.filter_objects(segvalues, 'person')
