"""Classify chart as a chart type."""

import os
from base64 import decodebytes
from io import BytesIO
from skimage.color import rgba2rgb
from skimage.transform import resize
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from gramex.data import filter as gfilter
from gramex.config import variables
import gramex.cache

OP = os.path
FOLDER = OP.dirname(__file__)

# fetch the label
charts = gfilter(variables['DB_URL'], table='charts')
to_keep = charts['validated_label'].value_counts()[
    charts['validated_label'].value_counts() > 21].index
charts = charts[charts['validated_label'].isin(to_keep)]
lenc = LabelEncoder()
lenc.fit_transform(charts['validated_label'])


def classifier(handler, model_name):
    """Classify a chart against a model.
    Args:
        handler (request object): gramex handler object
        model_name (str): model name to be loaded
    Returns:
        (dict): prediction label
    """
    # preprocess image
    param = handler.get_argument('image')
    image = param.split(',')[-1]
    image = decodebytes(image.encode())
    image = imread(BytesIO(image))

    image = rgba2rgb(image)
    image = resize(image, (224, 224), preserve_range=True)
    image = image.reshape((1, ) + image.shape)

    # load model and predict
    model_path = OP.join(FOLDER, 'assets', 'models', model_name)
    model = gramex.cache.open(model_path, keras.models.load_model)
    predictions = model.predict(image)

    labels = lenc.inverse_transform(predictions.argmax(axis=1))
    return {'label': labels}
