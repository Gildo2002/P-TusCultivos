import matplotlib.pyplot as plt
import re
import os
import numpy as np

from PIL import Image
from skimage.transform import resize

dirname = os.path.join(os.getcwd(), 'Pruebas')
imgpath = dirname + os.sep

images = []
directories = []
dircount = []
prevRoot=''
cant=0

print("leyendo imagenes de ",imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)

            image_resized = resize(image, (50, 50),anti_aliasing=True,clip=False,preserve_range=True)
            images.append(image_resized)

            img = Image.open(filepath)
            img = img.resize((100,100))
            name = 'Tizon - ' + str(cant) + '.jpg'
            img.save(name)


            #images.append(image)
            b = "Leyendo..." + str(cant)
            print (b, end="\r")
            if prevRoot !=root:
                print(root, cant)
                prevRoot=root
                directories.append(root)
                dircount.append(cant)
                cant=0
dircount.append(cant)
dircount = dircount[1:]
dircount[0]=dircount[0]+1
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))