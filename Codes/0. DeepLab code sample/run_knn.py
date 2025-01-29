import multiprocessing
import os
import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import fbeta_score
from image_processing import load_images
from utils import *

EXTRA_FEATURES_COUNT = 100
# kernels = load_pickle("kernels_{}".format(EXTRA_FEATURES_COUNT))
# if kernels is None:
kernels = load_pickle("kcws1")[0][:EXTRA_FEATURES_COUNT]
# dump_pickle(kernels, "kernels_kcws_{}".format(EXTRA_FEATURES_COUNT))

DATA_SIZE = 1_000_000

X_fs, y = load_images(os.path.join("labeled_data", "scans"),
                      os.path.join("labeled_data", "labels_corrected"),
                      threshold=DATA_SIZE,
                      resize_factor=4,
                      general_kernels_repeats=0,
                      edge_kernels_repeats=0,
                      include_raw_pixels=True,
                      kernels=kernels,
                      verbose=True)

# 1 main feature (raw pixel values) plus other Extra_Features
out_dir = os.path.join("output", str(X_fs.shape[1]))

# let's filter pores
# y[y == 5] = 0
# y[y > 0] = 1

# make unlabeled background
y[y == 0] = 5

fcmin = np.amin(X_fs, axis=0)
fcmax = np.amax(X_fs, axis=0)
dump_pickle(fcmin, "fcmin", dir=out_dir)
dump_pickle(fcmax, "fcmax", dir=out_dir)

# let's normalize our features :
X_fs = 2 * (X_fs - fcmin) / (fcmax - fcmin) - 1
# dump_pickle(X_fs, "X_fs", dir=out_dir)
# dump_pickle(y, "y", dir=out_dir)

N, F = X_fs.shape
L = N / 5

ind = np.arange(N)
np.random.shuffle(ind)
X_fs = np.array([X_fs[i] for i in ind])
y = np.array([y[i] for i in ind])
ind = None

knns2keep = 100
cpu_cores = multiprocessing.cpu_count()
test_sample_size = 300_000
test_batch = X_fs[:test_sample_size], y[:test_sample_size]
X_fs, y = X_fs[test_sample_size:], y[test_sample_size:]

labels = range(1, 6)

inds = np.arange(len(y))
inds = [inds[y == i] for i in labels]
# lbl_weights = np.array([len(ind) for ind in inds])
# lbl_weights = lbl_weights / np.sum(lbl_weights)


def make_random_sample_ids(size):
    factor = .5
    random_samples1 = [
        ind[rnd_i] for ind in inds for rnd_i in np.random.randint(
            0,
            high=len(ind),
            size=min(1 + int(factor * size / len(inds)), len(ind)))
    ]
    random_samples2 = np.random.randint(0,
                                        high=L,
                                        size=int(size * 1.1 -
                                                 len(random_samples1)))
    random_samples = np.concatenate([random_samples1, random_samples2])
    random_samples = np.unique(random_samples)
    return random_samples[:size]


def fit_knn(n_neighbors, train_sample_size, algorithm, weights, power, yy):

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        n_jobs=cpu_cores * 2,
        weights=weights,
        p=power,
    )

    random_samples = make_random_sample_ids(train_sample_size)
    knn.fit(X_fs[random_samples, :], yy[random_samples])

    t2 = time.time()
    X_, yy_ = test_batch
    # accu1 = knn.score(X_, yy_)
    # print("Accu score is {:.3%}, error={:.3%}".format(accu1, 1 - accu1),
    #       end="\t")
    yy_hat = knn.predict(X_)
    t2_end = time.time()
    print("It took {}s".format(t2_end - t2), end="\t")
    fb_scr_macro = fbeta_score(yy_, yy_hat, beta=1, average='macro')
    fb_scr_weighted = fbeta_score(yy_, yy_hat, beta=1, average='weighted')
    fb_scr_micro = fbeta_score(yy_, yy_hat, beta=1,
                               average='micro')  # same as accuracy
    #+ f1_score( yy_, yy_hat, average='weighted')
    print("F1 score is macro={:.3f} weighted={:.3f} micro={:.3f}".format(
        fb_scr_macro, fb_scr_weighted, fb_scr_micro))

    return knn, 1 - fb_scr_macro, t2_end - t2


best_knns = [
    load_pickle("best_knns2_0", default=[], dir=out_dir),
    load_pickle("best_knns2_1", default=[], dir=out_dir)
]
min_err = 10
min_predict_time = 200

stats = load_pickle("stats2", default={}, dir=out_dir)

kv_stats_list = []

for n_neighbors in [
        3,
        5,
        # 7,
        13,
        # 31,
]:
    for n_samples in [
            10,
            # 30,
            50,
            # 100,
            # 200,
            test_sample_size // 10,
            # test_sample_size // 6,
            # test_sample_size // 3,
    ]:
        if n_neighbors > n_samples: continue
        for power in range(2, 4):
            for algorithm in [
                    'brute',
                    'ball_tree',
                    # 'auto',
                    # 'kd_tree',
            ]:
                for weights in [
                        'uniform',
                        'distance',
                ]:
                    kv_stats = {
                        "n_neighbors": n_neighbors,
                        "n_samples": n_samples,
                        "p": power,
                        "algorithm": algorithm,
                        "weights": weights,
                    }
                    kv_stats_list.append(kv_stats)

for kv_stats in kv_stats_list:
    n_neighbors = kv_stats["n_neighbors"]
    n_samples = kv_stats["n_samples"]
    power = kv_stats["p"]
    algorithm = kv_stats["algorithm"]
    weights = kv_stats["weights"]
    print("KNN: {}".format(', '.join([
        "{}={}".format(k, v) for k, v in kv_stats.items()
        if k in ['n_samples', 'algorithm', 'n_neighbors', 'p', 'weights']
    ])))

    for i, yy in enumerate([
            # y2,
            y
    ]):
        dump_best_knns = False
        for _ in range(2):
            knn, err, predict_time = fit_knn(n_neighbors, n_samples, algorithm,
                                             weights, power, yy)

            for kv in kv_stats.items():
                if kv not in stats: stats[kv] = (0, 0, 0)
                cnt, sum_err, sum_pt = stats[kv]
                cnt += 1
                sum_err += err
                sum_pt += predict_time
                stats[kv] = cnt, sum_err, sum_pt

            min_err = min(err, min_err)
            min_predict_time = min(predict_time, min_predict_time)

            def reverse_score(args):
                err, predict_time = args[:2]
                return err / min_err  # + predict_time / min_predict_time

            if (len(best_knns[i]) < knns2keep or (reverse_score(
                [err, predict_time]) < reverse_score(best_knns[i][-1]))):
                print("updating best_knns_{} with ({:.2%} {:.2f}s)".format(
                    i, err, predict_time))
                best_knns[i].append((err, predict_time, knn))
                best_knns[i].sort(key=reverse_score)
                if len(best_knns[i]) >= knns2keep:
                    best_knns[i] = best_knns[i][:knns2keep]
                dump_best_knns = True

        if dump_best_knns:
            dump_pickle(best_knns[i], "best_knns2_" + str(i), dir=out_dir)

    dump_pickle(stats, "stats2", dir=out_dir)
