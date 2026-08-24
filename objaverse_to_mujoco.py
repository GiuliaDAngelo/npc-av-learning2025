#!/usr/bin/env python3
"""Convert Objaverse GLB assets into the CRIB/YCB MuJoCo layout:

  <out>/<category>_<k>/<category>_<k>.xml  (+ textured obj/mtl/png)

glTF is y-up; MuJoCo scenes here are z-up, so meshes are rotated +90 deg about X.
Meshes are merged to a single trimesh, centered, and scaled to a uniform extent.
Objects whose material has no texture fall back to a flat mean-color material.

  python objaverse_to_mujoco.py --src ~/DATA/objaverse_lvis --out ~/DATA/objaverse_lvis_mujoco
"""
import argparse
import glob
import os

import numpy as np
import trimesh

TARGET_EXTENT = 0.30

XML_TEMPLATE = """<mujoco model="{name}">
    <compiler angle="degree"/>
    <option><flag gravity="disable"/></option>
    <visual>
        <headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global offwidth="640" offheight="480"/>
    </visual>
    <asset>
{assets}
        <mesh name="{name}_mesh" file="{name}.obj"/>
    </asset>
    <worldbody>
        <light pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
        <body name="{name}" pos="0 0 0">
            <freejoint/>
            <geom type="mesh" mesh="{name}_mesh" {geom_mat}/>
        </body>
    </worldbody>
</mujoco>
"""


def convert(glb_path, name, out_root):
    dst = os.path.join(out_root, name)
    if os.path.exists(os.path.join(dst, f"{name}.xml")):
        return 'cached'
    scene = trimesh.load(glb_path)
    mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
    if mesh.vertices.shape[0] == 0:
        raise ValueError('empty mesh')

    # y-up (glTF) -> z-up, center, normalize scale
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    mesh.apply_translation(-mesh.bounding_box.centroid)
    ext = float(mesh.extents.max())
    mesh.apply_scale(TARGET_EXTENT / ext)

    os.makedirs(dst, exist_ok=True)
    has_tex = (hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None
               and getattr(mesh.visual, 'material', None) is not None
               and getattr(mesh.visual.material, 'baseColorTexture', None) is not None)
    if has_tex:
        mesh.visual.material.name = f'{name}_mtl'
        mesh.export(os.path.join(dst, f'{name}.obj'), include_texture=True)
        # MuJoCo reads the texture through the OBJ's mtl automatically only for
        # some builds; declare it explicitly for reliability.
        tex_file = None
        for f in os.listdir(dst):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                tex_file = f
                break
        if tex_file:
            assets = (f'        <texture name="{name}_tex" type="2d" file="{tex_file}"/>\n'
                      f'        <material name="{name}_material" texture="{name}_tex"/>')
            geom_mat = f'material="{name}_material"'
        else:
            has_tex = False
    if not has_tex:
        try:
            c = np.asarray(mesh.visual.to_color().vertex_colors)[:, :3].mean(0) / 255.0
        except Exception:
            c = np.array([0.6, 0.6, 0.6])
        mesh.export(os.path.join(dst, f'{name}.obj'))
        assets = (f'        <material name="{name}_material" '
                  f'rgba="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1"/>')
        geom_mat = f'material="{name}_material"'

    with open(os.path.join(dst, f'{name}.xml'), 'w') as f:
        f.write(XML_TEMPLATE.format(name=name, assets=assets, geom_mat=geom_mat))
    return 'textured' if has_tex else 'flat-color'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.expanduser('~/DATA/objaverse_lvis'))
    ap.add_argument('--out', default=os.path.expanduser('~/DATA/objaverse_lvis_mujoco'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stats = {}
    for cat in sorted(os.listdir(args.src)):
        glbs = sorted(glob.glob(os.path.join(args.src, cat, '*.glb')))
        for k, g in enumerate(glbs):
            name = f'{cat}_{k}'
            try:
                r = convert(g, name, args.out)
                stats[r] = stats.get(r, 0) + 1
            except Exception as e:
                print(f'  {name}: FAILED ({str(e)[:60]})')
                stats['failed'] = stats.get('failed', 0) + 1
    print('conversion:', stats)


if __name__ == '__main__':
    main()
