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


OP = os.path
FOLDER = OP.abspath(OP.dirname(__file__))


def classifier(handler, model_name, chart_db):
    """Classify a chart against a model.
    Args:
        handler (request object): gramex handler object
        model_name (str): model name to be loaded
        chart_db (str): sqlite path to database
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
    model = keras.models.load_model(model_path)
    predictions = model.predict(image)

    # fetch the label
    charts = gfilter(chart_db, table='charts')
    to_keep = charts['validated_label'].value_counts()[
        charts['validated_label'].value_counts() > 21].index
    charts = charts[charts['validated_label'].isin(to_keep)]

    lenc = LabelEncoder()
    lenc.fit_transform(charts['validated_label'])
    labels = lenc.inverse_transform(predictions.argmax(axis=1))

    return {'label': labels}