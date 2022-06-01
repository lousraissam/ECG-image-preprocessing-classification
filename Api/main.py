from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from numpy import argmax, asarray
from imageio import imread
import matplotlib.pyplot as plt
from skimage import color
from skimage.filters import threshold_otsu,gaussian
from skimage.transform import resize

import tensorflow as tf

MODEL = tf.keras.models.load_model("models\cnn_model_V4.h5")
CLASS_NAMES = ['History of MI', 'Myocardial Infarction', 'Normal ECG','Abnormal heartbeat']


app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",   #default port for react application
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    grayscale = color.rgb2gray(image)
    #smoothing image
    blurred_image = gaussian(grayscale, sigma=0.7)
    #thresholding to distinguish foreground and background
    #using otsu thresholding for getting threshold value
    global_thresh = threshold_otsu(blurred_image)
    #creating binary image based on threshold
    image = blurred_image > global_thresh
    #resize image
    image = resize(image, (224, 224))
    return image



@app.post("/predict")
async def predict(
    file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    prediction = (MODEL.predict(img_batch))
    # predictions = prediction[0]

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
    # class = ECG_CLASS[np.argmax(prediction)]
    # max_pred = np.max(predictions)*100
    # print(f'{} with {"%.2f" % max_pred}%  of confidence')

    # for pred in predictions:
    #     pred = pred*100
    #     pred = f' {"%.2f" % pred}x%'
    #     print(pred)
    
    

   



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port =8000)
