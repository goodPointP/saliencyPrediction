# Saliency Prediction
A project for Cognitive Science 3, 2021

Instance segmentation (for images)

HEATMAP SETTINGS

LOSS + OPTIMIZER

CLASSIFIER (LAST CNN LAYERS) SETUP

--> RUN CODE

TLDR on papers:
https://docs.google.com/document/d/1j7hx9PnsY-M0PakVxuyMlszC2uSiuI_3QFVEOQ7Q0bA/edit?usp=sharing


https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/

https://github.com/facebookresearch/detectron2

Questions
    - Which pretrained to use?


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

