#!/usr/bin/env python
import argparse

import numpy as np


def load_cat_file(fname):
    cats = []
    with open(fname) as f:
        for line in f:
            cats.append(line.strip().split()[-1])
    return cats


def load_inoutdoor(fname):
    d = {}
    with open(fname) as f:
        for line in f:
            category, env = line.strip().split()
            # originally: indoor 1, outdoor 2
            # modified: indoor 1, outdoor -1
            d[category[3:]] = 1 if env == '1' else -1
    return d


def detect_env(class_map, labels, pred_scores):
    weighted_score = sum([
        class_map[lbl] * sc for lbl, sc in zip(labels, pred_scores)]) / len(labels)

    return '[INDOOR]' if weighted_score > 0 else '[OUTDOOR]'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='prob2npy')
    parser.add_argument('-i', '--input-npy', type=str, required=True,
                        help='Path to prob features file .npy.')
    parser.add_argument('-c', '--labels', type=str, required=True,
                        help='Path to labels file.')
    parser.add_argument('-p', '--places-io', type=str, default=None,
                        help='Path to places indoor/outdoor labels file.')
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
        assert len(segment_index) == feats.shape[0], \
            "segment index line count does not match feats."

    assert feats.shape[1] == len(cats), ".npy file feature dim != # of labels"

    inout = None
    if args.places_io:
        inout = load_inoutdoor(args.places_io)
        assert len(inout) == feats.shape[1], \
            "indoor/outdoor file not compatible with feats."

    idxs = np.argpartition(-feats, kth=args.topk, axis=-1)[:, :args.topk]
    lines = []

    for i in range(idxs.shape[0]):
        scores = feats[i, idxs[i]]
        labels = [cats[j] for j in idxs[i]]

        # Add indoor/outdoor detection output
        inout_result = detect_env(inout, labels, scores) if inout else ['N/A']

        ordered = (sorted(zip(scores, labels), key=lambda x: x[0], reverse=True))

        # right-hand side, i.e. labels and their scores
        line = ' '.join(['{} ({:.4f})'.format(l, s) for (s, l) in ordered])

        # Add indoor/outdoor
        lines.append('{} {}'.format(inout_result, line))

    # Add segment IDs
    if segment_index:
        lines = ['{:15s} {}'.format(seg, line) for seg, line in zip(segment_index, lines)]

    # Dump
    for line in lines:
        print(line)
