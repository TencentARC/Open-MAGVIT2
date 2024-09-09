import numpy as np
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    return parser

def combine_npz(logdir):
    save_npzs = [npz for npz in os.listdir(logdir)]

    npzs = []

    for save_npz in save_npzs:
        tem_npz = np.load(os.path.join(logdir, save_npz))
        data = tem_npz["arr_0"]
        npzs.append(data)

    save_npz = np.vstack(npzs)
    np.random.shuffle(save_npz)
    np.savez(os.path.join(logdir, "sample.npz"), save_npz)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    combine_npz(args.logdir)