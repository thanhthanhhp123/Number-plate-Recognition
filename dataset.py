import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob
from shutil import copy
from skimage import io
import xml.etree.ElementTree as xet
import pytesseract as pt
from params import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array

path = glob(xml_path)
labels_dict = dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])
for filename in path:

    info = xet.parse(filename)
    root = info.getroot()
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)

    labels_dict['filepath'].append(filename)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)

df = pd.DataFrame(labels_dict)
df.to_csv('labels.csv',index=False)

def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('/content/Automatic-License-Plate-Detection/images',filename_image)
    return filepath_image
image_path = list(df['filepath'].apply(getFilename))
def get_dataset():
    labels = df.iloc[:,1:].values
    data = []
    output = []
    for ind in range(len(image_path)):
        image = image_path[ind]
        img_arr = cv2.imread(image)
        h,w,d = img_arr.shape
        load_image = load_img(image,target_size=(224,224))
        load_image_arr = img_to_array(load_image)
        norm_load_image_arr = load_image_arr/255.0

        xmin,xmax,ymin,ymax = labels[ind]
        nxmin,nxmax = xmin/w,xmax/w
        nymin,nymax = ymin/h,ymax/h
        label_norm = (nxmin,nxmax,nymin,nymax)
        # Append
        data.append(norm_load_image_arr)
        output.append(label_norm)
    X = np.array(data, dtype="float32")
    y = np.array(output, dtype="float32")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test