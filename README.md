# Saliency Prediction
A project for Cognitive Science 3, 2021

Troels' suggestion for parameters to use based on litterature:

Optimizer = Stochastic Gradient Descent (SGD) with learning rate set to 0.01 and divided by 10 when validation performance plateaus to max of 0.000001

Momentum = 0.9

Weight decay = 0.0005
    
Loss functions to try:

Deep fix uses pixel wise Euclidean loss

Linear combinations of many loss functions could be interesting and add some novelty

Weighted Binary Cross Entropy (W-BCE)

![image](https://user-images.githubusercontent.com/73380444/145030098-51c1215a-e24c-4781-a185-f88a2a52af46.png)

Focal loss:

![image](https://user-images.githubusercontent.com/73380444/145030459-025129a2-a15e-4eac-bf92-1a4c65c4e29e.png)

Normalized Scanpath Saliency (NSS):

![image](https://user-images.githubusercontent.com/73380444/145033055-59480580-b2cb-4028-8e2e-a5123d1df712.png)

Paper on loss functions for saliency models:

https://hal.inria.fr/hal-02264898/file/the_quest_for_the_loss_functionV1.pdf

Troels suggestions for structure of last layers:

Center bias can be added by adding regularization term to loss function, doesn't need a layer

![image](https://user-images.githubusercontent.com/73380444/145039139-cde963dd-861c-4e8e-9287-5bb95acb92da.png)

Inception block:
https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py
     

TLDR on papers:
https://docs.google.com/document/d/1j7hx9PnsY-M0PakVxuyMlszC2uSiuI_3QFVEOQ7Q0bA/edit?usp=sharing

https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/

https://github.com/facebookresearch/detectron2


Questions:

Which pretrained to use?

HEATMAP SETTINGS

LOSS + OPTIMIZER

CLASSIFIER (LAST CNN LAYERS) SETUP


Gaze Data model

BEFORE RUNNING CNN:
    - FIGURE PRETRAINED
    - FIGURE LOSS CRITERION
    - FIGURE CNN STRUCTURE
    - FIGURE I/O SHAPE&SIZE
        - USE DEEPFIX AS STARTING POINT
        
Questions

is this generative or predictive?????
    - predict heatmap
    - generate heatmap

which pretrained to use henceforth?
    -

normalize pixel values thusly increasing efficiency?
    - test this

different shape&size input?
    -
    
different shape&size output?
    - Deepfix paper read 
    
very carefully consider the gaussian function (and its params) in use. Right now just copy pasta, but has a lot of impact.
    - 
    
What kind of accuracy/loss criterion?
    - 


Combined model

How to combine

Do we feed it heatmap or fixation points?

Threshold for rendering?
    - 1 Pixel of object contained in edge of heatmap sufficient?

