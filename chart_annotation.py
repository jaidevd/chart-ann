from base64 import urlsafe_b64decode, decodebytes, urlsafe_b64encode
from gramex import data as gdata
import gramex.cache
from gramex.config import variables
from gramex.handlers import MLHandler, Capture
from gramex import service
import os
from tensorflow.keras.models import load_model
from io import BytesIO
from skimage.io import imread
from skimage.color import rgba2rgb
from skimage.transform import resize
import json
from sqlalchemy import create_engine
from sqlalchemy.exc import ArgumentError
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import regionprops
from skimage.measure import label
import segmentation as seg
from tornado.gen import coroutine, Return
from models import WindowObjectDetector

op = os.path
try:
    engine = create_engine(variables['COARSE_LABELS'])
except ArgumentError:
    pass

capture = Capture(engine='chrome')
MODEL_CACHE = {
    "_default": None
}

LABEL_ENCODER = ['barchart', 'scatterplot', 'treemap', 'choropleth']


def _cache_model(path):
    if not MODEL_CACHE['_default']:
        MODEL_CACHE['_default'] = WindowObjectDetector(load_model(path))
    return MODEL_CACHE['_default']


def view(handler, table="charts", pk="chart_id"):
    handler.set_header('Content-Type', 'image/png')
    handler.set_header('Content-Disposition', 'attachment; filename=image.png')
    data = gdata.filter(
        variables['COARSE_LABELS'], table=table,
        args={pk: [handler.path_args[0]]})
    url = data.iloc[0]['image'].split(',')[1]
    data = urlsafe_b64decode(url)
    return data


view_page = lambda handler: view(handler, 'pages', 'page_id')  # NOQA: E731


class ChartAnnModelHandler(MLHandler):
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
        return json.dumps(LABEL_ENCODER[prediction.argmax()])


def populate_annotations():
    vgg = _cache_model('vgg16-validated-five-classes.h5')
    """
    TODO

    1. for each row in charts table
    2. get the image
    3. update x, y, height, width columns in annotations table
    """
    df = gdata.filter(
            variables['COARSE_LABELS'], table='charts')
    for _, row in df.iterrows():
        x = imread(BytesIO(row['image'].split(',')[1]))
        pred, mask = seg.segment_image(
            x, vgg, blocksize=(224, 224), plot=False)
    rp = regionprops(mask.astype(int))
    mask = mask.astype(bool)
    labeled = label(mask)
    rp = regionprops(labeled)

    for region in rp:
        minrow, mincol, maxrow, maxcol = region.bbox
    """
    TODO: ...
    """


def plot_history(history, show=False):
    fig, ax = plt.subplots()
    for k, v in history.history.items():
        ax.plot(v, label=k)
    plt.legend()
    if show:
        plt.show()


def draw_grid(data, labels, size=6, figsize=(16, 16)):
    nrows = ncols = size
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i in range(size):
        for j in range(size):
            c = i * size + j
            try:
                ax[i, j].imshow(data[c])
                ax[i, j].axis('off')
                ax[i, j].set_title(labels[c])
            except IndexError:
                pass
    plt.tight_layout()


def get_labels():
    data = gramex.cache.query(
            "SELECT DISTINCT parent_label FROM charts",
            engine, state='SELECT COUNT(*) FROM charts')
    return data['parent_label'].unique()


def modify_completions(data, handler, args):
    """"""
    # ignore_cols = ['original_width', 'original_height']
    results = []
    for _, row in data.iterrows():
        results.append({
            "value": {
                "x": row.x,
                "y": row.y,
                "width": row.width,
                "height": row.height,
                "rotation": 0,
                "rectanglelabels": [
                  row.label
                ]
            },
            "id": row.annotation_id,
            "from_name": "tag",
            "to_name": "img",
            "type": "rectanglelabels"
            }
        )
    return json.dumps(results)


def update_label(handler):
    """
    pre-populated: update and delete options
    user input: insert
    """
    chart_id = handler.path_args[0]
    data = gramex.cache.query(
            "SELECT annotation_id FROM annotations WHERE chart_id={}".format(chart_id),
            engine, state='SELECT COUNT(*) FROM annotations')['annotation_id'].tolist()

    annotations = json.loads(handler.request.body.decode('utf8'))
    chart_ids = [item['id'] for item in annotations]

    # add incoming annotation that doesn't exist already
    to_delete = set(data) - set(chart_ids)
    [data.remove(k) for k in to_delete]
    # intersection operation
    to_update = set(data) & set(chart_ids)
    # add incoming annotation that doesn't exist already
    to_insert = set(chart_ids) - set(data)

    for annotation in annotations:
        value = annotation.pop('value')
        for key in 'x y width height'.split():
            annotation[key] = value[key]
        annotation['label'] = value['rectanglelabels'][0]
        annotation['chart_id'] = int(chart_id)
        annotation['annotation_id'] = annotation.pop('id')
    [k.update({'user': handler.current_user.email}) for k in annotations]

    df = pd.DataFrame.from_records(annotations)
    # to_delete
    gdata.delete(variables['COARSE_LABELS'], table="annotations",
                 id="annotation_id", args={'annotation_id': list(to_delete)})

    # to update
    for _id in to_update:
        args = df[df['annotation_id'] == _id]
        gdata.update(variables['COARSE_LABELS'], table="annotations", id="annotation_id",
                     args=args.to_dict(orient='list'))

    args = df[df['annotation_id'].isin(to_insert)]
    gdata.insert(variables['COARSE_LABELS'], table="annotations", id="annotation_id",
                 args=args.to_dict(orient='list'))


@coroutine
def process_screenshot(handler):
    model = _cache_model('vgg16-validated-five-classes.h5')
    url = handler.get_arg('url', False)
    if url:
        content = capture.png(url)
    else:
        content = handler.request.files['file'][0]['body']
    image = imread(BytesIO(content))
    annotation = yield service.threadpool.submit(seg.get_pre_annotations, image, model)
    meta = {}
    gramex.data.insert(
        variables['COARSE_LABELS'], table='pages', id='page_id', meta=meta,
        args={
            'image': ['data:image/png;base64,' + urlsafe_b64encode(content).decode('utf8')],
            'url': [url if url else handler.request.files['file'][0]['filename']],
        }
    )
    raise Return(json.dumps(dict(annotation=annotation, meta=meta)))
