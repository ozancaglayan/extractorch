#!/usr/bin/env python
import argparse

import numpy as np


def load_cat_file(fname):
    cats = []
    with open(fname) as f:
        for line in f:
            cats.append(line.strip().split()[-1])
    return cats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='prob2npy')
    parser.add_argument('-i', '--input-npy', type=str, required=True,
                        help='Path to prob features file .npy.')
    parser.add_argument('-c', '--labels', type=str, required=True,
                        help='Path to labels file.')
    parser.add_argument('-s', '--segment-index', type=str, default='',
                        help='Optional segment index file to concat to output.')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='Top-k items to dump.')

    # Parse arguments
    args = parser.parse_args()

    feats = np.load(args.input_npy)

    cats = load_cat_file(args.labels)
    segment_index = None

    if args.segment_index:
        segment_index = load_cat_file(args.segment_index)

    assert feats.shape[1] == len(cats), ".npy file feature dim != # of labels"

    idxs = np.argpartition(-feats, kth=args.topk, axis=-1)[:, :args.topk]

    lines = []

    for i in range(idxs.shape[0]):
        scores = feats[i, idxs[i]]
        labels = [cats[j] for j in idxs[i]]
        ordered = (sorted(zip(scores, labels), key=lambda x: x[0], reverse=True))

        lines.append(' '.join(['{} ({:.4f})'.format(l, s) for (s, l) in ordered]))

    if segment_index:
        lines = ['{} {}'.format(seg, line) for seg, line in zip(segment_index, lines)]

    for line in lines:
        print(line)

