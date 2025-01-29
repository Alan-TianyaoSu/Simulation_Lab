import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from scipy import ndimage

edge_kernels = [
    # Edge Detection1
    np.array([
        [0, -1, 0],  #
        [-1, 4, -1],
        [0, -1, 0],
    ]),
    np.array([
        [0, 0, -1, 0, 0],  #
        [0, 0, -1, 0, 0],
        [-1, -1, 8, -1, -1],
        [0, 0, -1, 0, 0],
        [0, 0, -1, 0, 0]
    ]),

    # Edge Detection2
    np.array([
        [-1, -1, -1],  #
        [-1, 8, -1],
        [-1, -1, -1]
    ]),

    # Bottom Sobel Filter
    np.array([
        [-1, -2, -1],  #
        [0, 0, 0],
        [1, 2, 1]
    ]),

    # Bottom Sobel Filter
    np.array([
        [-1, -2, -4, -2, -1],  #
        [0, -1, -2, -1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, 4, 2, 1]
    ]),
    np.array([
        [1, 2, 1],  #
        [0, 0, 0],
        [-1, -2, -1]
    ]),
    np.array([
        [1, 2, 4, 2, 1],  #
        [0, 1, 2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, -4, -2, -1]
    ]),

    # Left Sobel Filter
    np.array([
        [1, 0, -1],  #
        [2, 0, -2],
        [1, 0, -1]
    ]),

    # Right Sobel Filter
    np.array([
        [-1, 0, 1],  #
        [-2, 0, 2],
        [-1, 0, 1]
    ]),

    # Sharpen
    np.array([
        [0, -1, 0],  #
        [-1, 5, -1],
        [0, -1, 0]
    ]),
    # Emboss
    np.array([
        [-2, -1, 0],  #
        [-1, 1, 1],
        [0, 1, 2]
    ])
]

general_kernels = [
    # Box Blur
    (1 / 9.0) * np.array([
        [1, 1, 1],  #
        [1, 1, 1],
        [1, 1, 1]
    ]),
    # Gaussian Blur 3x3
    (1 / 16.0) * np.array([
        [1, 2, 1],  #
        [2, 4, 2],
        [1, 2, 1]
    ]),
    # Gaussian Blur 5x5
    (1 / 256.0) * np.array([
        [1, 4, 6, 4, 1],  #
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ]),
    # Unsharp masking 5x5
    -(1 / 256.0) * np.array([
        [1, 4, 6, 4, 1],  #
        [4, 16, 24, 16, 4],
        [6, 24, -476, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ]),
]


def make_features(
    X,
    label,
    random_kernel=None,
    include_raw_pixels=True,
    edge_kernels_repeats=2,
    general_kernels_repeats=1,
    random_kernels_repeats=0,
    kernels=None,
):
    random_kernels = None

    if kernels is not None:
        kernels = [(kernels, 1)]
    else:
        kernels = []

    if random_kernel is not None:
        size, count = random_kernel
        random_kernels = [(np.random.rand(size, size) * 1 - .5)
                          for _ in range(count)]
        # random_kernels = [k / np.sum(k) for k in random_kernels]

    kernels += [
        (edge_kernels, edge_kernels_repeats),
        (general_kernels, general_kernels_repeats),
    ]
    if random_kernels is not None:
        kernels.append((random_kernels, random_kernels_repeats))

    res = [X] if include_raw_pixels else []
    for ks, rpt in kernels:
        nth_pass = [X]
        executor = ThreadPoolExecutor()
        for _ in range(rpt):
            args_list = [(X_, k1) for X_ in nth_pass for k1 in ks]

            def task(args):
                X, kernel = args
                return ndimage.convolve(X, kernel, mode='reflect')

            next_pass = list(executor.map(task, args_list))
            res.extend(next_pass)
            nth_pass = next_pass

    res = np.array(res, dtype='uint8')
    res = res.transpose(1, 2, 0)
    res = res.reshape(-1, res.shape[2])
    label = label.reshape(-1)
    if random_kernels is not None:
        return res, label, random_kernels
    return res, label


def load_images(train_dir,
                label_dir,
                resize_factor=4,
                threshold=1_000_000,
                kernels=None,
                random_kernel=None,
                include_raw_pixels=True,
                edge_kernels_repeats=2,
                general_kernels_repeats=1,
                random_kernels_repeats=1,
                verbose=False):
    X, y = None, None
    res_kernels = None
    cnt = 1
    files = list(os.listdir(train_dir))
    random.shuffle(files)

    if verbose: t = time.time()
    for f in files:
        image = Image.open(os.path.join(train_dir, f))
        image = image.resize(
            (image.width // resize_factor, image.height // resize_factor),
            resample=Image.Resampling.NEAREST)
        image = np.array(image)
        label = Image.open(os.path.join(label_dir, f))
        label = label.resize(
            (label.width // resize_factor, label.height // resize_factor),
            resample=Image.Resampling.NEAREST)
        label = np.array(label)
        res = make_features(
            image,
            label,
            random_kernel=random_kernel,
            include_raw_pixels=include_raw_pixels,
            edge_kernels_repeats=edge_kernels_repeats,
            general_kernels_repeats=general_kernels_repeats,
            random_kernels_repeats=random_kernels_repeats,
            kernels=kernels,
        )
        if len(res) == 3:
            img_feats, labels, rnd_krnls = res
            if rnd_krnls is not None:
                if res_kernels is None:
                    res_kernels = list(rnd_krnls)
                else:
                    res_kernels.extend(list(rnd_krnls))
        else:
            img_feats, labels = res

        random_samples = np.random.randint(0, len(img_feats),
                                           threshold // len(files))
        X = np.concatenate(
            (X, img_feats[random_samples, :]
             )) if X is not None else img_feats[random_samples, :]
        y = np.concatenate((y, labels[random_samples]
                            )) if y is not None else labels[random_samples]
        # inds = np.arange(len(img_feats))
        # inds = [inds[labels == l] for l in range(1, 6)]
        # random_samples = [
        #     ind[rnd_i] for ind in inds for rnd_i in np.random.randint(
        #         0,
        #         high=len(ind),
        #         size=min(threshold // len(files) // len(inds), len(ind)))
        # ]
        # random_samples = np.array(random_samples)
        # X = np.concatenate((X, img_feats[random_samples]
        #                     )) if X is not None else img_feats[random_samples]
        # y = np.concatenate((y, labels[random_samples]
        #                     )) if y is not None else labels[random_samples]
        if verbose:
            print("#{} @ {}s".format(cnt, time.time() - t), X.shape)
        cnt += 1
        if X.shape[0] > threshold: break

    if res_kernels is None:
        return X, y

    return X, y, rnd_krnls


def load_image_features(
    path,
    resize_factor=4,
    include_raw_pixels=True,
    kernels=None,
):
    X = None

    X = Image.open(path)
    X = X.resize((X.width // resize_factor, X.height // resize_factor),
                 Image.Resampling.NEAREST)
    X = np.array(X)

    res = [X] if include_raw_pixels else []
    res.extend([ndimage.convolve(X, k1, mode='reflect') for k1 in kernels])

    res = np.array(res, dtype='uint8')
    res = res.transpose(1, 2, 0)

    return res


def load_image_labels(
    path,
    resize_factor=4,
):
    X = None

    X = Image.open(path)
    X = X.resize((X.width // resize_factor, X.height // resize_factor),
                 Image.Resampling.NEAREST)
    X = np.array(X)

    res = np.array(X, dtype='uint8')

    return res
