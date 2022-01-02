# Saliency Prediction
A project for Cognitive Science 3, 2021/22

The pipeline is controlled through the main.py file.

## Instructions for running the code locally
1. After cloning the repo, create a new conda env using
`conda create -n tf-gpu tensorflow-gpu`
2. Install the dependencies from requirements.txt
3. Download the mask_rcnn_coco.h5 from https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5 and place it inside the models/ folder
4. Specify path to the folder with the images you want to compress inside main.py, lines 13 and 17
5. Run the main.py
6. The compressed images will appear in the current directory
