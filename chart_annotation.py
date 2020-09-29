from base64 import urlsafe_b64decode, decodebytes
from gramex.data import filter as gfilter
from gramex.config import variables
from gramex.handlers import ModelHandler
import os
from tensorflow.keras.models import load_model
from io import BytesIO
from skimage.io import imread
from skimage.color import rgba2rgb
from skimage.transform import resize
import json
op = os.path

MODEL_CACHE = {
    "_default": None
}

LABEL_ENCODER = ['barchart', 'scatterplot', 'treemap', 'choropleth']


def _cache_model(path):
    if not MODEL_CACHE['_default']:
        MODEL_CACHE['_default'] = load_model(path)
    return MODEL_CACHE['_default']


def view(handler):
    handler.set_header('Content-Type', 'image/png')
    handler.set_header('Content-Disposition', 'attachment; filename=image.png')
    data = gfilter(variables['DB_URL'], table='charts', args={'chart_id': [handler.path_args[0]]})
    url = data.iloc[0]['image'].split(',')[1]
    data = urlsafe_b64decode(url)
    return data


class ChartAnnModelHandler(ModelHandler):
    """"""
    def prepare(self):
        self.set_header('Content-Type', 'application/json; charset=utf-8')
        if op.isfile(self.path):
            self.model_path = self.path
        elif op.isdir(self.path):
            if len(self.path_args):
                self.model_path = op.join(self.path, self.path_args[0])
            else:
                self.model_path = self.path
        ext = op.splitext(self.model_path)[-1]
        if ext == ".pkl":
            self.pickle_file_path = self.model_path
        elif ext == ".h5":
            self.keras_model_path = self.model_path

    def post(self):
        model = _cache_model(self.keras_model_path)
        self.img_shape = model.layers[0].input.shape[1:]
        self.write(self._predict(model))

    def _predict(self, model):
        '''Helper function for model.train.'''
        imgdata = self.request.body.decode('utf8').split(',')[-1]
        bytedata = decodebytes(imgdata.encode())
        height, width, channels = self.img_shape
        # image = Image.open(BytesIO(bytedata)).resize((width, height))
        # image = np.array(image)
        image = imread(BytesIO(bytedata))
        image = rgba2rgb(image)
        image = resize(image, (height, width), preserve_range=True)
        prediction = model.predict(image.reshape((1,) + image.shape)).ravel()
        print(prediction)
        return json.dumps(LABEL_ENCODER[prediction.argmax()])
