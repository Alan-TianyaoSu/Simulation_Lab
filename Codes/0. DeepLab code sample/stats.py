import os
import time
from sklearn.metrics import fbeta_score

import numpy as np

from image_processing import load_images
from utils import dump_pickle, load_pickle

EXTRA_FEATURES_COUNT = 100
kernels = load_pickle("kcws1")[0][:EXTRA_FEATURES_COUNT]

# print(" ".join(["{:.3f}".format(scores[i]) for i in range(17)]))
# print(" ".join(["{:.3f}".format(weights[i]) for i in range(17)]))
# print(" ".join(["{:.3f}".format(scores[i] * weights[i]) for i in range(17)]))

DATA_SIZE = 5_000_000

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

stats = load_pickle("stats2", dir=out_dir)
best_knns = load_pickle("best_knns2_0", dir=out_dir)
fcmin = load_pickle("fcmin", dir=out_dir)
fcmax = load_pickle("fcmax", dir=out_dir)
X_fs = (X_fs - fcmin) / (fcmax - fcmin)

# let's filter pores
# y[y == 5] = 0
# y[y > 0] = 1

# make unlabeled background
y[y == 0] = 5

# X_fnrm = load_pickle("X_fnrm", dir=out_dir)
# y = load_pickle("y", dir=out_dir)

# idcs = list(
#     np.arange(
#         start=0,
#         stop=len(X_fnrm),
#         step=(len(X_fnrm) // (len(X_fnrm) // 10_000)),
#     ))

print("---------------------------")
print("------- Overal Stats ------")
print("---------------------------")
min_err = min(sum_err / cnt for _, (cnt, sum_err, _) in stats.items())
min_time = min(sum_t2 / cnt for _, (cnt, _, sum_t2) in stats.items())


def score_stat(stat):
    err, time = stat[:2]
    return min_err / err + min_time / time


print(" Error \t\tPredict Time \t  key, value")
stats_arr = [(sum_err / cnt, sum_t2 / cnt, k, v)
             for (k, v), (cnt, sum_err, sum_t2) in stats.items()]
for row in sorted(stats_arr, key=score_stat, reverse=True):
    print("{:.4%} \t{:.3f}s \t\t {}, {}".format(*row))

print("------------------------------")
print("-------Testing Best KNNs------")
print("------------------------------")
knn_stats = []
for _, _, knn in reversed(best_knns):
    kv_stats = [("n_samples", len(knn._fit_X))] + list(
        knn.get_params().items())
    print("KNN: {}".format(', '.join([
        "{}={}".format(k, v) for k, v in kv_stats
        if k in ['n_samples', 'algorithm', 'n_neighbors', 'p', 'weights']
    ])),
          end="\t",
          flush=True)
    t = time.time()

    # accu = sum(
    #     knn.score(X_fnrm[i1:i2], y[i1:i2])
    #     for i1, i2 in zip(idcs[:-1], idcs[1:]))
    # accu /= len(idcs) - 1

    # accu = knn.score(X_fs, y)
    yy_hat = knn.predict(X_fs)

    t_end = time.time()

    fb_scr_macro = fbeta_score(y, yy_hat, beta=1, average='macro')
    fb_scr_weighted = fbeta_score(y, yy_hat, beta=1, average='weighted')
    fb_scr_micro = fbeta_score(y, yy_hat, beta=1,
                               average='micro')  # same as accuracy

    print("`predict` took {:.2f}s".format(t_end - t), end='\t')
    print("F1 score is macro={:.3f} weighted={:.3f} micro={:.3f}".format(
        fb_scr_macro, fb_scr_weighted, fb_scr_micro))

    err = 1 - fb_scr_micro
    print("Error percentage {:.2%}".format(err), flush=True)

    knn_stats.append((err, t_end - t, kv_stats, fb_scr_macro, fb_scr_weighted,
                      fb_scr_micro, knn))

min_err = min(stat[0] for stat in knn_stats)
min_time = min(stat[1] for stat in knn_stats)


def score_knn_stat(stat):
    err, time = stat[:2]
    return min_err / err + min_time / time


print("-------------------------------")
print("------- Overal KNN Stats ------")
print("-------------------------------")
print("Error \t\tPredict Time\tAttributes")
knn_stats = sorted(knn_stats, key=score_knn_stat, reverse=True)
for err, t, kvs, _ in knn_stats:
    print("{:.4%} \t{:.2f}s \t {}".format(
        err, t, ', '.join([
            "{}={}".format(k, v) for k, v in kvs
            if k in ['n_samples', 'algorithm', 'n_neighbors', 'p', 'weights']
        ])))

dump_pickle(knn_stats, "knn_stats", dir=out_dir)
