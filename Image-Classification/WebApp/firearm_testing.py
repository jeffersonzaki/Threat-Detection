import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Image Classes
classes = ['Assault Rifle', 'Handgun']


# Prediction Function
def predict(model, path):
    img = load_img(path, target_size=(300, 300))  # Loading image
    img = img_to_array(img)  # Transforming image to array
    img = img / 255  # Normalizing Image
    img = np.expand_dims(img, axis=0)  # Expanding dimensions
    predict = model.predict(img)  # Predicting the image
    pred_name = classes[np.argmax(predict)]  # Predicting the name
    prediction = str(round(predict.max() * 100, 2))
    return prediction + '%', pred_name


if __name__ == "__predict__":
    predict()
