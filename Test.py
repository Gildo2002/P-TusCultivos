from posixpath import sep, split
from numpy.testing._private.utils import print_assert_equal
from skimage.transform import resize
from skimage.io import imsave
import matplotlib.pyplot as plt
import re
import os
import numpy as np

from pathlib import Path
from PIL import Image

#Subidas de imagenes para el modelo

dirname = os.path.join(os.getcwd(), '/media/hexlinux/ROM/TusCultivos/P-TusCultivos/test/')
dirNewImg = '/media/hexlinux/ROM/TusCultivos/P-TusCultivos/ImgResize/'
imgpath = dirname + os.sep

nameDir = []

for content in os.listdir(imgpath):
  for i in os.listdir(imgpath + content):
    nameDir.append(imgpath + content + '/' + i + '/')

nameImg = []

for content in os.listdir(imgpath):
  for i in os.listdir(imgpath + content):
    nameImg.append(content + '_' + i + '_')

for name in nameImg:
  print(name)
  pass

images = []
prevRoot=''
cant=0


for name in nameDir:
   #print(name,end='\n')
   pass

error = []

for filenames in nameDir:
    cant = 0
    value = filenames.split('/')
    newdir = os.path.join(dirNewImg, value[-4], value[-3], value[-2])
    print("Transformando en " + filenames)
    os.makedirs(newdir)
    for filename in os.listdir(filenames):
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
          cant=cant+1
          filepath = os.path.join(filenames, filename)
          #print(value[-2] + '_' + str(cant))
          try:
            image = plt.imread(filepath,0)
            newImg = os.path.join(newdir,value[-2] + '_' + str(cant))
            new_img = resize(image, (200,200),anti_aliasing=True,clip=False,preserve_range=True)
            imsave(newImg + '.jpg', new_img)
          except:
            error.append(new_img)
            continue


print(error)