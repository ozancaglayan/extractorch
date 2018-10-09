#!/usr/bin/env python
import sys
import argparse
from collections import Counter

import numpy as np


def load_index_file(fname):
    cats = []
    with open(fname) as f:
        for line in f:
            cats.append(line.strip().split('/')[0])
    return cats, [c[:11] for c in cats]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='feat2meanpool')
    parser.add_argument('-i', '--input-npy', type=str, required=True,
                        help='Path to prob features file .npy.')
    parser.add_argument('-c', '--index', type=str, required=True,
                        help='Index .txt file with image names in order.')
    parser.add_argument('-k', '--top-k', type=int, default=0,
                        help='Number of top labels for BoW features.')
    parser.add_argument('-o', '--out-prefix', type=str, required=True,
                        help='Output prefix for saved files.')

    # Parse arguments
    args = parser.parse_args()

    # Load features
    feats = np.load(args.input_npy)

    if args.top_k > 0 and feats.shape[1] not in (1000, 365):
        print('Top-k only useful if features are classification scores.')
        sys.exit(1)

    segs, vids = load_index_file(args.index)

    count_vids = Counter(vids)
    count_imgs = Counter(segs)
    count_segs = Counter([v[:11] for v in count_imgs.keys()])

    st = 0
    vid_ranges = []
    for vid, n_segs in count_vids.items():
        vid_ranges.append((vid, slice(st, st + n_segs)))
        st += n_segs

    st = 0
    seg_ranges = []
    for seg, n_imgs in count_imgs.items():
        seg_ranges.append((seg, slice(st, st + n_imgs)))
        st += n_imgs

    out_shape = (len(count_imgs), feats.shape[1])
    per_video = np.zeros(out_shape, dtype='float32')
    per_seg = np.zeros(out_shape, dtype='float32')

    if args.top_k > 0:
        per_video_topk = np.zeros(out_shape, dtype='uint8')
        per_seg_topk = np.zeros(out_shape, dtype='uint8')

    #####################
    # 1 feature per video
    #####################
    st = 0
    for vid, rng in vid_ranges:
        pool = feats[rng].mean(0)
        rng = slice(st, st + count_segs[vid])
        per_video[rng] = pool
        st += count_segs[vid]

        if args.top_k > 0:
            # Get top-k range
            topk = pool.argsort()[-args.top_k:]
            per_video_topk[rng, topk] = 1

    st = 0
    #######################
    # 1 feature per segment
    #######################
    for idx, (_, rng) in enumerate(seg_ranges):
        pool = feats[rng].mean(0)
        per_seg[idx] = pool

        if args.top_k > 0:
            # Get top-k range
            topk = pool.argsort()[-args.top_k:]
            per_seg_topk[idx, topk] = 1

    np.save('{}-pv.npy'.format(args.out_prefix), per_video.astype('float16'))
    np.save('{}-ps.npy'.format(args.out_prefix), per_seg.astype('float16'))
    if args.top_k > 0:
        np.save('{}-pv-top{}.npy'.format(args.out_prefix, args.top_k), per_video_topk)
        np.save('{}-ps-top{}.npy'.format(args.out_prefix, args.top_k), per_seg_topk)
