"""
1. Write a python script to read images from a folder and classify them to different folders using a Tensor Flow JS ML model hosted remotely.
"""

import os
import requests
import json
import shutil




# Path to the folder containing the images to be classified
path = "/Users/kedar/Desktop/nativa-box/walnuts_ML_AI/SimpleImageClassification/SiteVisits/xyzFoods"

# Path to the folder where the classified images will be stored
destination = "/Users/kedar/Desktop/nativa-box/walnuts_ML_AI/SimpleImageClassification/SiteVisits/xyzFoods"

# URL of the Tensor Flow JS ML model
url = "https://teachablemachine.withgoogle.com/models/lcuBVP0R1/"

# Iterate through the images in the folder
for filename in os.listdir(path):
    # Read the image
    print(filename)    
    with open(path + "/" + filename, "rb") as image_file:
        image = image_file.read()
    # Send the image to the model for classification
    print (url)
    r = requests.post(url, data=image)
    # Get the classification result

    result = json.loads(r.text)
    # Get the label with the highest probability
    label = result["classifications"][0]["className"]
    # Create a folder with the label name if it doesn't exist
    if not os.path.exists(destination + "/" + label):
        os.makedirs(destination + "/" + label)
    # Move the image to the folder
    shutil.move(path + "/" + filename, destination + "/" + label + "/" + filename)