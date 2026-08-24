#!/usr/bin/env python3
"""Record raw per-saccade exploration data for offline experiments.

Runs reference-mode episodes (bottom-up saccades, no RL) and saves one .npz per
episode containing the full saccade stream: fixation patches, image coordinates,
rotation quaternions, and encoder embeddings. This is the corpus for the encoder
bake-off and offline graph construction (see notes/BACKGROUND.md).

Usage:
  MUJOCO_GL=egl python record_explorations.py --num_objects 40 --episodes 1 --steps 240
"""
import argparse
import os
import random

import numpy as np
import cv2


def pad_patch(p, size):
    """Saccades near the frame border yield clipped crops; zero-pad to a uniform size."""
    if p.shape[:2] == (size, size):
        return p
    out = np.zeros((size, size, p.shape[2]), dtype=p.dtype)
    out[:p.shape[0], :p.shape[1]] = p
    return out

# Headless-safe: no GUI needed for recording
for fn in ('namedWindow', 'resizeWindow', 'imshow', 'destroyWindow'):
    setattr(cv2, fn, lambda *a, **k: None)
cv2.waitKey = cv2.waitKeyEx = lambda *a, **k: -1

import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--objects', nargs='+', default=None, help='Object names; default: random sample')
    ap.add_argument('--objects_dir', default=None,
                    help='Object library root (default: CRIB, via train.OBJECTS_DIR)')
    ap.add_argument('--num_objects', type=int, default=40, help='How many objects to sample if --objects not given')
    ap.add_argument('--episodes', type=int, default=1, help='Episodes per object')
    ap.add_argument('--steps', type=int, default=240, help='Saccade steps per episode')
    ap.add_argument('--patch_source', default='rgb', choices=['rgb', 'events'])
    ap.add_argument('--saliency_source', default='events', choices=['events', 'rgb', 'itti'])
    ap.add_argument('--out', default='explorations')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--randomize', action='store_true',
                    help='Randomize initial pose and rotation speeds per episode so '
                         'repeat episodes of one object explore it differently')
    args = ap.parse_args()

    random.seed(args.seed)
    if args.objects_dir:
        train.OBJECTS_DIR = args.objects_dir
    train.PATCH_SOURCE = args.patch_source
    train.SALIENCY_SOURCE = args.saliency_source
    train.RECORD_SACCADES = True
    train.MAX_REF_STEPS = args.steps

    if args.objects is None:
        all_objs = sorted(os.listdir(train.OBJECTS_DIR))
        objects = random.sample(all_objs, min(args.num_objects, len(all_objs)))
    else:
        objects = args.objects

    os.makedirs(args.out, exist_ok=True)
    print(f"Recording {args.episodes} episode(s) x {len(objects)} objects, "
          f"{args.steps} steps, patches={args.patch_source}, saliency={args.saliency_source}")

    for obj in objects:
        xml = os.path.join(train.OBJECTS_DIR, obj, f"{obj}.xml")
        if not os.path.exists(xml):
            print(f"  skip {obj}: no xml")
            continue
        for ep in range(args.episodes):
            path = os.path.join(args.out, f"{obj}_ep{ep}.npz")
            if os.path.exists(path):
                print(f"  {obj} ep{ep}: already recorded, skipping")
                continue
            if args.randomize:
                train.INITIAL_YAW_ANGLE = random.uniform(0, 2 * np.pi)
                train.INITIAL_PITCH_ANGLE = random.uniform(-0.4, 0.4)
                train.INITIAL_YAW_SPEED = random.choice([-1, 1]) * random.uniform(0.03, 0.07)
                train.INITIAL_PITCH_SPEED = random.uniform(-0.02, 0.02)
            train.run_simulation(xml, f"{obj}_{ep}", mode='reference')
            log = train.LAST_SACCADE_LOG
            if not log:
                print(f"  {obj} ep{ep}: no saccades recorded, skipping")
                continue
            np.savez_compressed(
                path,
                obj=obj,
                patch_source=args.patch_source,
                saliency_source=args.saliency_source,
                patches=np.stack([pad_patch(s['patch'], train.ROI_SIZE) for s in log]),
                coords=np.array([s['coord'] for s in log], dtype=np.float32),
                quats=np.array([s['quat'] for s in log], dtype=np.float32),
                embeddings=np.stack([s['embedding'] for s in log]).astype(np.float32),
            )
            print(f"  {obj} ep{ep}: {len(log)} saccades -> {path}")


if __name__ == '__main__':
    main()
