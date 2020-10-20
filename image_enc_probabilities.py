# coding: utf-8
import pandas as pd
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import numpy as np
import h5py
from skimage.transform import resize
from tqdm import tqdm
from skimage.color import rgba2rgb
from skimage.io import imread
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt

ROOT = './gramexcharts/'

X = []
for f in tqdm(os.listdir(ROOT)):
    im = imread(os.path.join(ROOT, f))
    if im.shape[-1] > 3:
        im = rgba2rgb(im)
    im = resize(im, (224, 224))  # NOQA: E912
    X.append(im)

X = np.array(X)

with h5py.File('thumbnails.h5', 'w') as fout:
    fout.create_dataset('thumbnails', data=X)

model = load_model('vgg16-validated-five-classes.h5')

encoder = Model(model.layers[1].input, model.layers[1].output)
x = imread('/tmp/choropleth.png')
x = rgba2rgb(x)  # if n_channels is > 3
x = resize(x, (224, 224))  # NOQA: E912
xAct = encoder(x)

# Get nearest neighbors
thumbnailsEncoded = encoder(X)
nn = NearestNeighbors(n_neighbors=10, n_jobs=-1)
nn.fit(thumbnailsEncoded)

y = nn.kneighbors(xAct, return_distance=False)

tsne = TSNE(n_components=2)
thumb_reduced = tsne.fit_transform(thumbnailsEncoded.T)
thumb_reduced = tsne.fit_transform(tf.transpose(thumbnailsEncoded))

xtest = X
w_red = thumb_reduced

fig, ax = plt.subplots(figsize=(12, 12))
for i, vec in enumerate(xtest):
    left, bottom = w_red[i]
    right, top = w_red[i] + 0.5
    ax.imshow(vec, extent=[left, right, bottom, top], cmap=plt.cm.gray)


ax.axis([w_red[:, 0].min() - 1, w_red[:, 0].max() + 1,
         w_red[:, 1].min() - 1, w_red[:, 1].max() + 1])


# Probabilities from VGG16 predictions
preds = model.predict(X)
prob = tf.nn.softmax(preds, axis=1)
x = prob.numpy()
xdf = pd.DataFrame(x, columns=['barchart', 'donut', 'map', 'multiline', 'scatterplot'])
sns.heatmap(xdf, cmap=plt.cm.viridis)
