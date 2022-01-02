#%%
from math import log10, sqrt
import cv2
import numpy as np
import os
  
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  
# def main():
#      original = cv2.imread("original_image.png")
#      compressed = cv2.imread("compressed_image.png", 1)
#      value = PSNR(original, compressed)
#      print(f"PSNR value is {value} dB")

dir_name = '../../Datasets/HighResSet/OURFINALOUTPUTS/'
# Get list of all files only in the given directory
list_of_files = filter( lambda x: os.path.isfile(os.path.join(dir_name, x)),
                        os.listdir(dir_name) )
# Create a list of files in directory along with the size
files_with_size = [ (file_name, os.stat(os.path.join(dir_name, file_name)).st_size) 
                    for file_name in list_of_files  ]
#%%

ourfinalvalue = []
targetedfinalvalue = []
for file_name, size in files_with_size:
    #print(size, ' -->', file_name) 
    original = cv2.imread("../../Datasets/HighResSet/ORIGINALIMAGES/"+file_name.split('.jpg')[0].split('final-')[1]+'.png')
    compressed = cv2.imread("../../Datasets/HighResSet/OURFINALOUTPUTS/"+file_name, 1)
    compressedTargeted = cv2.imread("../../Datasets/HighResSet/TARGETEDFILESIZES/"+file_name.split('final-')[1], 1)
    ourfinalvalue.append(PSNR(original, compressed))
    targetedfinalvalue.append(PSNR(original, compressedTargeted))
    
# %%
ourMeanPSNR = np.array(ourfinalvalue).mean()
targetMeanPSNR = np.array(targetedfinalvalue).mean()
# %%
