import gzip
import os
import pickle


def load_pickle(name, dir=".", default=None):
    try:
        with gzip.open(os.path.join(dir, name + ".pkl.gz"), "rb") as fp:
            return pickle.load(fp)
    except Exception as e:
        print(e)

    try:
        with open(os.path.join(dir, name + ".pkl"), "rb") as fp:
            return pickle.load(fp)
    except Exception as e:
        print(e)

    return default


def dump_pickle(obj, name, dir=".", compress=True):
    os.makedirs(dir, exist_ok=True)
    if compress:
        try:
            with gzip.open(os.path.join(dir, ".unnamed.tmp"), "wb") as fp:
                pickle.dump(obj, fp)
            try:
                os.rename(os.path.join(dir, name + ".pkl.gz"),
                          os.path.join(dir, ".unnamed.pkl.tmp"))
            except:
                pass
            os.rename(os.path.join(dir, ".unnamed.tmp"),
                      os.path.join(dir, name + ".pkl.gz"))
            try:
                os.remove(os.path.join(dir, ".unnamed.pkl.tmp"))
            except:
                pass
        except:
            pass
    else:
        try:
            with open(os.path.join(dir, ".unnamed.tmp"), "wb") as fp:
                pickle.dump(obj, fp)
            try:
                os.rename(os.path.join(dir, name + ".pkl.gz"),
                          os.path.join(dir, ".unnamed.pkl.tmp"))
            except:
                pass
            os.rename(os.path.join(dir, ".unnamed.tmp"),
                      os.path.join(dir, name + ".pkl"))
            try:
                os.remove(os.path.join(dir, ".unnamed.pkl.tmp"))
            except:
                pass
        except:
            pass
