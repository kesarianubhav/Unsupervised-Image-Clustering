import numpy as np
import tensorflow as tf
import scipy as sci
import matplotlib.image as img
import matplotlib.pyplot as plt
import os , sys ,cv2
from PIL import Image as pimg
import matplotlib.pyplot as plt
from neupy import algorithms, environment
environment.reproducible()

plt.style.use('ggplot')
train_data=[]
train_data2 =[]
missing_file=[]
x_pad = 5
y_pad = 5
#count = 0
def show():
    for i in train_data:
        i=i/255
        print("i's shape ="+str(i.shape))
        image_array = np.array( i )
        image_flatten=np.reshape(i,(50*50*3,1))
        train_data2.append(image_flatten)
    for i in train_data2:
        print("i's shape ="+str(i.shape))
        #print(i)


def file_size(fname):
    stat=os.stat("baseball2/"+str(fname))
    return stat.st_size


def reader():
    images_path = "../F-ML/baseball2"
    files = [f for f in os.listdir(images_path) ]
    print("Working with {0} images".format(len(files)))
    print("Image examples: ")
    print(len(files))
    for i in files:
        print(i)
        if (file_size(i)!=0):
            image = pimg.open("baseball2/"+str(i))
            image=image.resize((50,50),pimg.ANTIALIAS)
            #print(image.shape)
            arr=np.array(image)
            #print(arr.shape)
            train_data.append(arr)
            print(len(train_data))
        else:
            missing_file.append(i)

if __name__ =="__main__":
    reader()
    show()
    input_data = np.array(train_data2)

    sofmnet = algorithms.SOFM(
        n_inputs=7500,
        n_outputs=100,

        step=0.5,
        show_epoch=20,
        shuffle_data=True,
        verbose=True,

        learning_radius=0,
        features_grid=(10, 10),
    )

    plt.plot(input_data.T[0:1, :], input_data.T[0:1, :], 'ko')
    sofmnet.train(input_data, epochs=100)

    print("> Start plotting")
    plt.xlim(-1, 1.2)
    plt.ylim(-1, 1.2)

    plt.plot(sofmnet.weight[0:1, :], sofmnet.weight[1:2, :], 'bx')
    plt.show()

    for data in input_data:
        print(sofmnet.predict(np.reshape(data, (2, 1)).T))

