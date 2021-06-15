import matplotlib.pyplot as plt
import re
import os
import numpy as np
import keras

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from PIL import Image
from skimage.transform import resize

#Subidas de imagenes para el modelo

dirname = os.path.join(os.getcwd(), '/media/hexlinux/ROM/TusCultivos/P-TusCultivos/Enfermedades de plantas/Cultivos Granos')
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

            #image_resized = resize(image, (50, 50),anti_aliasing=True,clip=False,preserve_range=True)
            #images.append(image_resized)

            images.append(image)
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
dircount[0] = dircount[0] + 1

print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))

#Creacion de las etiquetas y las clases

labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
print("numeros de etiquetas creadas: ",len(labels))

plantas=[]
indice=0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice , name[len(name)-1])
    plantas.append(name[len(name)-1])
    indice=indice+1

y = np.array(labels)
X = np.array(images, dtype=np.uint8) 

# Numeros unicos de las etiquetas del entrenamiento
classes = np.unique(y)
nClasses = len(classes)
print('Numero total de la salida : ', nClasses)
print('Clases de salida : ', classes)

# mezclar todo y crear los grupos de entrenamiento y testing

train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)
print('Perfil de datos de entrenamiento : ', train_X.shape, train_Y.shape)
print('Perfil de datos de testeo : ', test_X.shape, test_Y.shape)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# display the change for category label using one-hot encoding
print('Etiqueta original:', train_Y[0])
print('Despues de la conversion de one-hot:', train_Y_one_hot[0])

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

INIT_LR = 1e-3
epochs = 6
batch_size = 64

sport_model = Sequential()
sport_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(21,28,3)))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(MaxPooling2D((2, 2),padding='same'))
sport_model.add(Dropout(0.5))

sport_model.add(Flatten())
sport_model.add(Dense(32, activation='linear'))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(Dropout(0.5)) 
sport_model.add(Dense(nClasses, activation='softmax'))

sport_model.summary()

sport_model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adagrad(lr=INIT_LR, decay=INIT_LR / 100),metrics=['accuracy'])

sport_train_dropout = sport_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

#salvamos la red, para reutilizarla en el futuro, sin tener que volver a entrenarla
sport_model.save("plants.h5py")

test_eval = sport_model.evaluate(test_X, test_Y_one_hot, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])