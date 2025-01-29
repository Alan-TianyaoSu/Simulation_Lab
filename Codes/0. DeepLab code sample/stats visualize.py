import os
import time

import numpy as np
from PIL import Image

from image_processing import load_image_features, general_kernels, edge_kernels
from utils import load_pickle

EXTRA_FEATURES_COUNT = 0
kernels = load_pickle("kernels_{}".format(EXTRA_FEATURES_COUNT))

files = list(os.listdir(os.path.join("labeled_data", "scans")))
# 1 main feature (raw pixel values) plus other Extra_Features
knn_dir = os.path.join("output", str(EXTRA_FEATURES_COUNT + 16))
fcmin = load_pickle("fcmin", dir=knn_dir)
fcmax = load_pickle("fcmax", dir=knn_dir)

colors = {
    1: [0, 1, 0],
    2: [1, 0, 1],
    3: [0.929, 0.624, 0.125],
    4: [1, 0, 0],
    5: [0, 0.4471, 0.7412],
}
colors[0] = [0, 0, 0]
for k in colors.keys():
    row = np.array(colors[k]) * 256
    row = np.minimum(row, 255)
    row = np.maximum(row, 0)
    row = np.array(row, np.uint8)
    colors[k] = row

best_knns = load_pickle("best_knns_0", dir=knn_dir)
for i, (_, _, knn) in enumerate(best_knns):
    if i != 11: continue

    out_dir = os.path.join("output_2", "y_hat_{}".format(i),
                           str(EXTRA_FEATURES_COUNT + 1))
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        X_feats = load_image_features(
            os.path.join("labeled_data", "scans", f),
            resize_factor=2,
            include_raw_pixels=True,
            kernels=kernels,
        )

        W, H, _ = X_feats.shape

        X_feats = X_feats.reshape(W * H, -1)
        X_fnrm = (X_feats - fcmin) / (fcmax - fcmin)

        t = time.time()
        y_hat = knn.predict(X_fnrm)
        t_end = time.time()
        print("`predict` took {:.2f}s".format(t_end - t))

        y_hat = y_hat.reshape(W, H)

        im = Image.fromarray(y_hat, mode="L")
        im.save(os.path.join(out_dir, f))

        img = np.zeros((W, H, 3), np.uint8)
        for l, col in colors.items():
            img[y_hat == l, :] = col
        im = Image.fromarray(img, mode="RGB")
        im.save(os.path.join(out_dir, "{}_colored.png".format(f)))
