from email.mime import application
from unittest import result
import cv2
import os
from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename


model_path='model.h5'
model=load_model(model_path)
application = Flask(__name__)



diseases_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

labels=['nv','mel','bkl','bcc','akiec','vasc','df']

@application.route('/')
def man():
    return render_template('index.html')


@application.route('/predict', methods=['POST'])
def home():
    if request.method == 'POST':
        f = request.files['file']
        sfname = '../sample/'+str(secure_filename(f.filename))
        f.save(sfname) 
        f=cv2.imread(sfname)
        f=cv2.resize(f,(32,32))
        f=f.reshape(-1,32,32,3)
        p=model.predict([f])
        x=np.argmax(p)
        os.remove(sfname)
        return render_template('index.html',result= diseases_dict[labels[x]])
       
   



if __name__ == "__main__":
    application.run(debug=True)