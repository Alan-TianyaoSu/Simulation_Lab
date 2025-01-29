import os

import numpy as np
from PIL import Image
from scipy import ndimage

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

kernel = [[1] * 3 for _ in range(3)]
resize_factor = 1
path = os.path.join("labeled_data", "labels")
for f in list(os.listdir(path)):
    label = Image.open(os.path.join(path, f))
    label = label.resize(
        (label.width // resize_factor, label.height // resize_factor))
    y = np.array(label)

    dds = [(di, dj) for di in range(-1, 2) for dj in range(-1, 2)
           if not di == dj == 0]

    label_cnts = [
        ndimage.convolve(np.where(y == l, 1, 0), kernel, mode='reflect')
        for l in range(1, 6)
    ]
    label_cnts = np.array(label_cnts, dtype="<i1")
    label_cnts = label_cnts.transpose(1, 2, 0)

    y = np.where(y == 0, label_cnts.argmax(axis=-1) + 1, y)

    w, h = y.shape
    # y = y.reshape(-1)

    img = np.zeros((w, h, 3), np.uint8)

    for k, v in colors.items():
        img[y == k, :] = v

    # img = img.reshape(w, h, 3)

    im = Image.fromarray(img, mode="RGB")
    im.save("out_y_{}.png".format(f))

    os.makedirs(os.path.join("labeled_data", "labels_corrected"),
                exist_ok=True)
    im = Image.fromarray(y, mode="L")
    im.save(os.path.join("labeled_data", "labels_corrected", f))
