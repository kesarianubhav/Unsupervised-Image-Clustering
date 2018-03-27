
# Unsupervised Clustering of Visually Similar Images using a Neural Network

#My Effort -
#To cluster the images , firstly we should be able to extract the features from the Images
#There can be various ways to extract the features from the image eg. edge detection , SURF etc .

# I have used the pre-trained model of the VGG-16 Convolutional Neural Network for extracting the features
# From the (Last -1) layer I am extracting the features and flattening it to become of the shape (1,25088)

# Now I have fed those flattened features into the clustering algorithm to cluster
# Clustering Algorithm Used = K means

# And I also did tried to cluster it using the Self Organizing Featuer Maps (SOFM) ,
# but the result was not satisfatory

import numpy as np
#import tensorflow as tf
import scipy as sci
import matplotlib.image as img
import matplotlib.pyplot as plt
import os , sys ,cv2
from PIL import Image as pimg
import matplotlib.pyplot as plt
from neupy import algorithms, environment
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as hac
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train_data=[]
missing_file=[]
files=[]


X_train=[]
clusters={}
#n=0

def file_size(fname):
    stat=os.stat("f/"+str(fname))
    return stat.st_size


def vgg16():
    n=0
    model = VGG16(weights='imagenet', include_top=False)
    images_path = "../Task/f"
    for f in os.listdir(images_path):
        files.append(f)
    for img_path in files:
        if(file_size(img_path)!=0):
            print(str(img_path))
            n+=1
            print(n)

            img = image.load_img("f/"+str(img_path), target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model.predict(x)
            #print(features.shape)
            a=features.shape[0]
            b=features.shape[1]
            c=features.shape[2]
            d=features.shape[3]
            features = np.array(features)
            features= features.flatten()
            X_train.append(features)
            features = (features - np.mean(features))/(np.std(features,axis=0));


            #print(features)
    #MinMaxScalar(copy = False , feature_range=(0,1))

def kmeans():
        #print("X_train's length="+str(len(X_train)))
        #print(X_train)
        kmeans = KMeans(n_clusters=15, init='k-means++', max_iter=100, n_init=10,tol=1e-6, verbose=1)
        kmeans.fit(X_train)
        labels=kmeans.predict(X_train)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        print ("centroids : ")
        #print (centroids)
        print ("labels : ")
        #print (labels)
        p=dict(zip(files,labels))

        for i,j in p.items() :
            new_path=str("../Task/"+str(j))
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            img = cv2.imread("f/"+str(i), 1)
            cv2.imwrite(os.path.join(new_path , i), img)

def som():
    input_data = np.array(X_train)
    sofmnet = algorithms.SOFM(
    n_inputs=25088,
    n_outputs=25,

    step=0.5,
    show_epoch=100,
    shuffle_data=True,
    verbose=True,

    learning_radius=0,
    features_grid=(5,5),
    )

    sofmnet.train(input_data,epochs=100);
    plt.plot(input_data.T[0:1, :], input_data.T[1:2, :], 'ko')

    print("> Start plotting")
    plt.xlim(-1, 1.2)
    plt.ylim(-1, 1.2)

    plt.plot(sofmnet.weight[0:1, :], sofmnet.weight[1:2, :], 'bx')
    plt.show()

    for data in input_data:
        print(sofmnet.predict(np.reshape(data, (2, 1)).T))



if __name__ =="__main__":
    #reader()
    vgg16()
    #kmeans();
    #som();
    hierarchial()
