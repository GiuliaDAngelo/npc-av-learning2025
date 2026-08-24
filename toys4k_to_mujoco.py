#!/usr/bin/env python3
"""Convert Toys4K blend files into the CRIB/YCB MuJoCo layout.

Two stages per object: headless Blender exports the .blend to GLB (textures packed),
then objaverse_to_mujoco.convert() produces the textured-OBJ + wrapper XML. Output
objects are named <category>_<k> in the shared layout, so every recorder/eval script
works unchanged via --objects_dir.

  python toys4k_to_mujoco.py --categories mug cup bowl --max_per 10
  python toys4k_to_mujoco.py            # everything (4k objects; hours)
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from objaverse_to_mujoco import convert

SRC = os.path.expanduser('~/DATA/Toys4k/toys4k_blend_files')
OUT = os.path.expanduser('~/DATA/Toys4k/mujoco_objects')


def blend_to_glb(blend, glb):
    expr = (f"import bpy; bpy.ops.export_scene.gltf(filepath={glb!r}, "
            f"export_format='GLB')")
    # Distro blender runs on system python, which has no numpy (the glTF
    # exporter needs it); borrow the venv's via PYTHONPATH.
    venv_sp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '.venv', 'lib', 'python3.14', 'site-packages')
    env = {**os.environ, 'PYTHONPATH': venv_sp}
    r = subprocess.run(['blender', '-b', blend, '--python-use-system-env',
                        '--python-expr', expr],
                       capture_output=True, text=True, timeout=180, env=env)
    if not os.path.exists(glb):
        raise RuntimeError((r.stdout + r.stderr)[-200:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--categories', nargs='+', default=None, help='default: all')
    ap.add_argument('--max_per', type=int, default=10)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cats = args.categories or sorted(os.listdir(SRC))
    stats = {}
    for cat in cats:
        blends = sorted(glob.glob(os.path.join(SRC, cat, '*', '*.blend')))[:args.max_per]
        for k, blend in enumerate(blends):
            name = f'{cat}_{k}'
            if os.path.exists(os.path.join(args.out, name, f'{name}.xml')):
                stats['cached'] = stats.get('cached', 0) + 1
                continue
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    glb = os.path.join(tmp, 'o.glb')
                    blend_to_glb(blend, glb)
                    r = convert(glb, name, args.out)
                stats[r] = stats.get(r, 0) + 1
            except Exception as e:
                print(f'  {name}: FAILED ({str(e)[:80]})')
                stats['failed'] = stats.get('failed', 0) + 1
        print(f'{cat}: done ({len(blends)} instances)')
    print('conversion:', stats)


if __name__ == '__main__':
    main()
