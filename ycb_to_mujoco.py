#!/usr/bin/env python3
"""Convert YCB google_16k scans into the same MuJoCo layout as CRIB Data:

  YCB Data/mujoco_objects/<name>/<name>.xml   (+ textured.obj/.mtl, texture_map.png)

Textured OBJs load natively in MuJoCo (UV coords + PNG texture). Meshes are scaled to
a uniform extent so the pipeline's default camera framing matches the CRIB objects.
Leading YCB index digits are stripped from names (011_banana -> banana; letter variants
keep a suffix: 065-a_cups -> cups_a).

  python ycb_to_mujoco.py --tgz_dir /media/matt/bigdata/DATA/YCB/tgz --out "YCB Data/mujoco_objects"
"""
import argparse
import glob
import os
import re
import shutil
import tarfile
import tempfile

import numpy as np

TARGET_EXTENT = 0.30  # max bounding-box side after scaling (matches CRIB framing)

XML_TEMPLATE = """<mujoco model="{name}">
    <compiler angle="degree"/>

    <option>
        <flag gravity="disable"/>
    </option>

    <visual>
        <headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global offwidth="640" offheight="480"/>
    </visual>

    <asset>
        <texture name="{name}_tex" type="2d" file="texture_map.png"/>
        <material name="{name}_material" texture="{name}_tex"/>
        <mesh name="{name}_mesh" file="textured.obj" scale="{s} {s} {s}"/>
    </asset>

    <worldbody>
        <light pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
        <body name="{name}" pos="0 0 0">
            <freejoint/>
            <geom type="mesh" mesh="{name}_mesh" material="{name}_material"/>
        </body>
    </worldbody>
</mujoco>
"""


def clean_name(raw):
    m = re.match(r'^\d+(?:-([a-z]))?[_-](.+)$', raw)
    if not m:
        return raw
    letter, base = m.group(1), m.group(2).replace('-', '_')
    return f"{base}_{letter}" if letter else base


def obj_extent(path):
    ext = None
    vs = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                vs.append([float(x) for x in line.split()[1:4]])
    v = np.array(vs)
    return float((v.max(0) - v.min(0)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tgz_dir', default='/media/matt/bigdata/DATA/YCB/tgz')
    ap.add_argument('--out', default='YCB Data/mujoco_objects')
    args = ap.parse_args()

    tgzs = sorted(glob.glob(os.path.join(args.tgz_dir, '*_google_16k.tgz')))
    print(f"{len(tgzs)} archives")
    ok = 0
    for tgz in tgzs:
        raw = os.path.basename(tgz).replace('_google_16k.tgz', '')
        name = clean_name(raw)
        dst = os.path.join(args.out, name)
        if os.path.exists(os.path.join(dst, f"{name}.xml")):
            ok += 1
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with tarfile.open(tgz) as tf:
                    tf.extractall(tmp, filter='data')
                src = os.path.join(tmp, raw, 'google_16k')
                if not os.path.isdir(src):
                    hits = glob.glob(os.path.join(tmp, '**', 'textured.obj'), recursive=True)
                    if not hits:
                        print(f"  {raw}: no textured.obj, skipping")
                        continue
                    src = os.path.dirname(hits[0])
                os.makedirs(dst, exist_ok=True)
                for f in ('textured.obj', 'textured.mtl', 'texture_map.png'):
                    shutil.copy(os.path.join(src, f), os.path.join(dst, f))
            scale = TARGET_EXTENT / obj_extent(os.path.join(dst, 'textured.obj'))
            with open(os.path.join(dst, f"{name}.xml"), 'w') as f:
                f.write(XML_TEMPLATE.format(name=name, s=f"{scale:.6f}"))
            ok += 1
        except Exception as e:
            print(f"  {raw}: FAILED ({e})")
            shutil.rmtree(dst, ignore_errors=True)
    print(f"converted {ok}/{len(tgzs)} objects -> {args.out}")


if __name__ == '__main__':
    main()
