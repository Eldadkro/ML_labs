# -*- coding: utf-8 -*-
"""
Image Processing
"""
#Import required libraries
import numpy as np
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg
from PIL import Image

#Open Image
im0 = Image.open("sad_cat.jpg")
title0 = Image.open("titles.jpg")
im0.show()
title0.show()

img = mpimg.imread('sad_cat.jpg')
title = mpimg.imread('titles.jpg')
title1 = title[:,:,0]

rows = title1.shape[0]
cols = title1.shape[1]

plt.imshow(img)
plt.show()

plt.imshow(title)
plt.show()

plt.imshow(title1, cmap = 'gray')
plt.show()

gr_im =  0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]
plt.imshow(gr_im, cmap = 'gray')
plt.show()

title_inv = 255 - title1
plt.imshow(title_inv, cmap = 'gray')
plt.show()

rows2=rows//2
title2 = title_inv[0:rows2,:]
plt.imshow(title2, cmap = 'gray')
plt.show()
        

