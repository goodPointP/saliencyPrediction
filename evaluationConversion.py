#!/usr/local/bin/python3
#%%
import io
import math
import sys
import numpy as np
from PIL import Image


def JPEGSaveWithTargetSize(im, filename, target):
   """Save the image as JPEG with the given name at best quality that makes less than "target" bytes"""
   # Min and Max quality
   Qmin, Qmax = 10, 96
   # Highest acceptable quality found
   Qacc = -1
   while Qmin <= Qmax:
      m = math.floor((Qmin + Qmax) / 2)

      # Encode into memory and get size
      buffer = io.BytesIO()
      im.save(buffer, format="JPEG", quality=m)
      s = buffer.getbuffer().nbytes

      if s <= target:
         Qacc = m
         Qmin = m + 1
      elif s > target:
         Qmax = m - 1

   # Write to disk at the defined quality
   if Qacc > -1:
      im.save(filename, format="JPEG", quality=Qacc)
   else:
      print("ERROR: No acceptble quality factor found for image "+filename, file=sys.stderr)

################################################################################
# main
################################################################################

# Load sample image
#im = Image.open('lena.png')

#JPEGSaveWithTargetSize(im, "test.jpg", 50000)

# Save at best quality under 100,000 bytes

# %%

# READ OUR RESULTING filesizes
import os
dir_name = '../../Datasets/EVALUATIONSUBSET/OURFINALOUTPUTS/'
# Get list of all files only in the given directory
list_of_files = filter( lambda x: os.path.isfile(os.path.join(dir_name, x)),
                        os.listdir(dir_name) )
# Create a list of files in directory along with the size
files_with_size = [ (file_name, os.stat(os.path.join(dir_name, file_name)).st_size) 
                    for file_name in list_of_files  ]
# Iterate over list of files along with size 
# and print them one by one.
for file_name, size in files_with_size:
    #print(size, ' -->', file_name) 
    im = Image.open("../../Datasets/EVALUATIONSUBSET/ORIGINALIMAGES/"+file_name)
    JPEGSaveWithTargetSize(im, "../../Datasets/EVALUATIONSUBSET/TARGETEDFILESIZES/"+file_name+".jpg", size)
# %%
