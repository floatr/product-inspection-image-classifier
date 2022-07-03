"""
1. Write a python script to read images from a folder
2. Classify every image by using the tensorflow.js model hosted on Teachable machine
3. Move the classified image to a subfolder with the same class name.
"""

import os
import shutil
import requests

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from numpy import loadtxt

# Load the model
model = load_model('/Users/kedar/Desktop/nativa-box/walnuts_ML_AI/SimpleImageClassification/product-inspection-image-classifier/MLModel/keras_model.h5')
labels_file = open('/Users/kedar/Desktop/nativa-box/walnuts_ML_AI/SimpleImageClassification/product-inspection-image-classifier/MLModel/labels.txt','r')
labels = []
labels = [line.split() for line in labels_file]
print (labels)

# Path to the folder containing the images to be classified
path = "/Users/kedar/Desktop/nativa-box/walnuts_ML_AI/SimpleImageClassification/SiteVisits/hira-30062022"

# Path to the folder where the classified images will be moved
destination = "/Users/kedar/Desktop/nativa-box/walnuts_ML_AI/SimpleImageClassification/SiteVisits/hira-30062022"

# URL of the tensorflow.js model hosted on Teachable machine
#url = "https://teachablemachine.withgoogle.com/models/lcuBVP0R1/model.json"


# Iterate through all the images in the folder
for filename in os.listdir(path):    

    if not filename.startswith('.') and os.path.isfile(os.path.join(path, filename)) :

        # Read the image
        with open(os.path.join(path, filename), "rb") as image_file:
            print(image_file)            

            # Create the array of the right shape to feed into the keras model
            # The 'length' or number of images you can put into the array is
            # determined by the first position in the shape tuple, in this case 1.
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            # Replace this with the path to your image
            image = Image.open(image_file)
            #resize the image to a 224x224 with the same strategy as in TM2:
            #resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)

            #turn the image into a numpy array
            image_array = np.asarray(image)
            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            # Load the image into the array
            data[0] = normalized_image_array
            
            # run the inference
            prediction = model.predict(data)        

            #sense check probability predictions
            probabilites = prediction * 100
            print (probabilites)

            max_index = np.argmax(prediction,axis=1)
            print(max_index)

            class_name = labels[max_index[0]]
            print(image_file,class_name)            

            os.makedirs(os.path.join(destination, class_name[1]), exist_ok=True)
            # Move the image to the subfolder with the same class name
            shutil.move(os.path.join(path, filename), os.path.join(destination, class_name[1], filename))
            #input()