# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:01:07 2020

@author: seknayu
"""
from keras.models import load_model
import numpy as np
from keras.preprocessing import image,backend

class Animal:
    def __init__(self,filename):
        self.filename = filename
    def predictionAnimal(self):
        # load model
        backend.clear_session()
        model = load_model('model.h5')
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
        print(result)
        if result[0][0] == 1:
            prediction = 'cat'
            return [{ "image" : prediction}]
        elif result[0][1] == 1:
            prediction = 'dog'
            return [{ "image" : prediction}]
        elif result[0][2] == 1:
            prediction = 'monkey'
            return [{ "image" : prediction}]
        else:
            prediction = 'squirrel'
            return [{ "image" : prediction}]

