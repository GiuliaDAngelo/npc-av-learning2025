import numpy as np 
import cv2 
from attention import *
import torch
import SEMDRunner 
import mujoco

def quat_mul(q2, q1):
    # q = q2 ⊗ q1  (apply q1, then q2); q is [w,x,y,z]
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1
    ], dtype=np.float64)

def quat_conj(q):
    w,x,y,z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)

def quat_norm(q):
    return q / (np.linalg.norm(q) + 1e-12)

def cam_basis_world_from_xmat(data, cam_id=0):
    R = np.asarray(data.cam_xmat[cam_id]).reshape(3, 3)
    right_w = R[:, 0]        # image right in world
    up_w    = R[:, 1]        # image up in world
    fwd_w   = -R[:, 2]       # camera forward in world
    right_w = right_w / (np.linalg.norm(right_w) + 1e-12)
    up_w    = up_w    / (np.linalg.norm(up_w) + 1e-12)
    fwd_w   = fwd_w   / (np.linalg.norm(fwd_w) + 1e-12)
    return right_w, up_w, fwd_w

def axis_world_to_axis_img(axis_world, right_w, up_w):
    # camera basis is "image right" and "image up" in WORLD coords
    ax = float(np.dot(axis_world, right_w))   # +x pixel is right
    ay = float(np.dot(axis_world, up_w))      # +ay is "up" in camera sense

    # convert to pixel coordinates (y down):
    ay = -ay

    n = np.hypot(ax, ay)
    if n < 1e-9:
        return None
    return (ax / n, ay / n)

def draw_divider_line(img_bgr, axis_img, color=(0,0,0), thickness=2):
    if axis_img is None:
        return img_bgr
    H, W = img_bgr.shape[:2]
    ax, ay = float(axis_img[0]), float(axis_img[1])
    n = np.hypot(ax, ay)
    if n < 1e-9:
        return img_bgr
    ax /= n; ay /= n
    cx = 0.5*(W-1); cy = 0.5*(H-1)
    L = int(2 * max(H, W))
    p1 = (int(cx - ax*L), int(cy - ay*L))
    p2 = (int(cx + ax*L), int(cy + ay*L))
    cv2.line(img_bgr, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    return img_bgr


def halfplane_mask_blocks(nz, block, H, W, axis_img, center_xy=None, side_sign=+1):
    if axis_img is None:
        return np.zeros_like(nz, dtype=bool)

    ax, ay = axis_img  # must be numbers now
    ax = float(ax); ay = float(ay)

    if center_xy is None:
        cx = 0.5*(W-1); cy = 0.5*(H-1)
    else:
        cx, cy = float(center_xy[0]), float(center_xy[1])

    hh, ww = nz.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    bx = (xx + 0.5) * block
    by = (yy + 0.5) * block

    s = ax * (by - cy) - ay * (bx - cx)
    return (s * float(side_sign) > 0) & nz

def quat_from_axis_angle(axis, ang_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = np.sin(0.5*ang_rad)
    return np.array([np.cos(0.5*ang_rad), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float64)  # wxyz

def cam_basis_world(data, cam_id=0):
    # MuJoCo camera orientation matrix in world coords
    R = data.cam_xmat[cam_id].reshape(3,3)
    right = R[:,0]                         # image right (world)
    up    = R[:,1]                         # image up (world)
    fwd   = -R[:,2]                        # camera forward (world), OpenGL looks along -z_cam
    right = right / (np.linalg.norm(right)+1e-12)
    up    = up    / (np.linalg.norm(up)+1e-12)
    fwd   = fwd   / (np.linalg.norm(fwd)+1e-12)
    return right, up, fwd

def pixel_to_ray_cam(u, v, W, H, fovy_deg):
    cx = 0.5*(W-1); cy = 0.5*(H-1)
    x_ndc = (u - cx) / cx
    y_ndc = (v - cy) / cy  # down is +
    fovy = np.deg2rad(fovy_deg)
    tan_y = np.tan(0.5*fovy)
    tan_x = tan_y * (W / H)

    # camera coords: +x right, +y up, -z forward
    r_cam = np.array([x_ndc*tan_x, -y_ndc*tan_y, -1.0], dtype=np.float64)
    return r_cam / (np.linalg.norm(r_cam)+1e-12)

def signed_angle(a, b, axis):
    axis = axis / (np.linalg.norm(axis)+1e-12)
    a_p = a - axis*np.dot(a, axis)
    b_p = b - axis*np.dot(b, axis)
    a_p = a_p / (np.linalg.norm(a_p)+1e-12)
    b_p = b_p / (np.linalg.norm(b_p)+1e-12)
    s = np.dot(axis, np.cross(a_p, b_p))
    c = np.dot(a_p, b_p)
    return np.arctan2(s, c)

def yaw_pitch_to_center(u, v, W, H, fovy_deg):
    r = pixel_to_ray_cam(u, v, W, H, fovy_deg)
    fwd = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    up  = np.array([0.0, 1.0,  0.0], dtype=np.float64)
    right = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # yaw about up to remove horizontal error (xz-plane)
    r_xz = np.array([r[0], 0.0, r[2]], dtype=np.float64)
    f_xz = np.array([fwd[0], 0.0, fwd[2]], dtype=np.float64)
    r_xz = r_xz / (np.linalg.norm(r_xz)+1e-12)
    f_xz = f_xz / (np.linalg.norm(f_xz)+1e-12)
    yaw = signed_angle(r_xz, f_xz, up)

    # rotate ray by yaw, then pitch about right to match forward
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[ cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]], dtype=np.float64)
    r2 = Ry @ r
    pitch = signed_angle(r2, fwd, right)

    # correction is opposite direction
    return -yaw, -pitch  # radians

def delta_quat_world_from_pixel(model, data, u, v, W, H, cam_id=0,
                                gain=1.0, deadband_px=2.0, max_step_deg=10.0):
    cx = 0.5*(W-1); cy = 0.5*(H-1)
    if abs(u - cx) < deadband_px and abs(v - cy) < deadband_px:
        return np.array([1.0,0.0,0.0,0.0], dtype=np.float64), 0.0, 0.0

    fovy_deg = float(model.cam_fovy[cam_id])
    dyaw, dpitch = yaw_pitch_to_center(u, v, W, H, fovy_deg)
    dyaw *= gain
    dpitch *= gain

    max_step = np.deg2rad(max_step_deg)
    dyaw = float(np.clip(dyaw, -max_step, max_step))
    dpitch = float(np.clip(dpitch, -max_step, max_step))

    right_w, up_w, _ = cam_basis_world(data, cam_id=cam_id)

    q_yaw   = quat_from_axis_angle(up_w,    dyaw)
    q_pitch = quat_from_axis_angle(right_w, dpitch)
    q_delta = quat_norm(quat_mul(q_pitch, q_yaw))  # yaw then pitch
    return q_delta, dyaw, dpitch

def quaternion_from_axis_angle(axis, ang_rad):
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    s = np.sin(ang_rad/2.0)
    return np.array([np.cos(ang_rad/2.0), ax[0]*s, ax[1]*s, ax[2]*s], dtype=np.float64)  # [w,x,y,z]

def set_pitch_roll_quat(data, pitch_deg, roll_deg):
    # local X then local Z: q_total = q_roll ⊗ q_pitch
    q_pitch = quaternion_from_axis_angle([1,0,0], np.deg2rad(pitch_deg))
    q_roll  = quaternion_from_axis_angle([0,0,1], np.deg2rad(roll_deg))
    q_total = quat_mul(q_roll, q_pitch)
    data.qpos[3:7] = q_total  # free joint quat [w,x,y,z]

def delta_pitch_roll_quat(pitch_deg, roll_deg):
    q_pitch = quaternion_from_axis_angle([1,0,0], np.deg2rad(pitch_deg))
    q_roll  = quaternion_from_axis_angle([0,0,1], np.deg2rad(roll_deg))
    return quat_mul(q_roll, q_pitch)  # roll ⊗ pitch

def get_total_quaternion(pitch_deg, roll_deg, base_q):
    delta = delta_pitch_roll_quat(pitch_deg, roll_deg)   # relative rotation
    q_total = quat_mul(base_q,delta)                   
    return q_total / (np.linalg.norm(q_total) + 1e-12)

def to_unitary_ssp(vec, n=None):
    v = vec.astype(np.float64).ravel()
    if n is None: n = v.size
    V = np.fft.rfft(v, n=n)
    V_unit = V / (np.abs(V) + 1e-9)        # unit magnitude per bin
    u = np.fft.irfft(V_unit, n=n)
    u /= (np.linalg.norm(u) + 1e-9)         # optional
    return u

def edges_to_heat(edge_map, out_h, out_w, clip_p=99):
            # edge_map: float/int 2D
    m = edge_map.astype(np.float32)
    vmax = float(np.percentile(m, clip_p))
    if vmax <= 1e-9:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    m = np.clip(m / vmax, 0.0, 1.0)
    u8 = (m * 255).astype(np.uint8)
    heat = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return heat

def saliency_from_events(ev_bgr, net_attention, device, config):
    # ev_bgr is the colored event frame (H,W,3) from EventFrameRenderer
    gray = cv2.cvtColor(ev_bgr, cv2.COLOR_BGR2GRAY)  # uint8 (H,W)
    H, W = gray.shape
    ten = torch.from_numpy(np.ascontiguousarray(gray)).float().unsqueeze(0)  # (1,H,W) in 0..255
    ten = ten.to(device)
    S, salmax_coords = run_attention(ten, net_attention, device, (H,W), (H,W), config.ATTENTION_PARAMS['num_pyr'])
    
    S = S.astype(np.float32)
    m = S.max()
    if m <= 1e-12:
        S8 = np.zeros((H, W), dtype=np.uint8)
    else:
        S8 = (S / m * 255.0).clip(0, 255).astype(np.uint8)

    Scolor = cv2.applyColorMap(S8, cv2.COLORMAP_JET)
    return Scolor, S, salmax_coords
    

def make_foveated_events_binary(ev_img, x, y, radius = 38 , tile=8):
    H, W = ev_img.shape[:2]
    if ev_img.ndim == 2:
        gray = ev_img
    else:
        gray = cv2.cvtColor(ev_img, cv2.COLOR_BGR2GRAY)
    # ON events: anything > 0
    on = (gray > 0).astype(np.uint8)
    small_h = H // tile
    small_w = W // tile
    # manual max pooling 
    on_small = np.zeros((small_h, small_w), dtype=np.uint8)
    for i in range(small_h):
        for j in range(small_w):
            patch_on = on[i*tile:(i+1)*tile, j*tile:(j+1)*tile]
            on_small[i, j] = patch_on.max()
    # upsample with nearest-neighbor (keeps sharp binary look)
    on_low = cv2.resize(on_small, (W, H), interpolation=cv2.INTER_NEAREST)
    # periphery: 0 = no event (black), 255 = event (white)
    periphery = np.zeros((H, W), dtype=np.uint8)
    periphery[on_low == 1] = 255
    periphery = cv2.cvtColor(periphery, cv2.COLOR_GRAY2BGR)
    # foveal region = full resolution events
    Y, X = np.ogrid[:H, :W]
    mask = (X - x)**2 + (Y - y)**2 <= radius**2
    foveated = periphery.copy()
    foveated[mask] = ev_img[mask]
    return foveated

def make_foveated_events_eccentricity(ev_img, x, y, fovea_radius=40, n_rings=4, max_tile=8):
    H, W = ev_img.shape[:2]
    gray = ev_img if ev_img.ndim == 2 else cv2.cvtColor(ev_img, cv2.COLOR_BGR2GRAY)
    on = (gray > 0).astype(np.uint8) * 255

    Y_grid, X_grid = np.mgrid[:H, :W]
    dist = np.sqrt((X_grid - x)**2 + (Y_grid - y)**2)

    d_max = np.sqrt(max(x, W - x)**2 + max(y, H - y)**2)

    # Eq. 2: geometric ring edges — R^c_i = R^c_{i-1} + R^c_{i-1}/2
    ring_edges = [fovea_radius]
    r = fovea_radius
    for _ in range(n_rings):
        r = r * 1.5
        ring_edges.append(min(r, d_max))

    # Eq. 1: tile size scales linearly with distance from fovea
    # R^s(x) = -(max_tile / d_fovea) * x + max_tile
    # x here is ring center distance from fovea edge
    d_fovea = ring_edges[-1] - fovea_radius
    tiles = []
    for i in range(n_rings):
        ring_center_dist = (ring_edges[i] + ring_edges[i + 1]) / 2.0 - fovea_radius
        tile = max_tile - (max_tile / d_fovea) * (d_fovea - ring_center_dist)
        tiles.append(max(1, round(tile)))

    _palette = [
        (255, 255, 255),
        (0, 255, 255),
        (255, 0, 255),
        (0, 165, 255),
        (0, 255, 0),
        (0, 0, 255),
        (128, 0, 128),
        (255, 128, 0),
    ]
    colors = _palette[:n_rings + 1]

    pooled = {}
    for tile in set(tiles) | {max_tile}:  # ensure max_tile is always precomputed
        sh, sw = H // tile, W // tile
        sm = cv2.resize(on, (sw, sh), interpolation=cv2.INTER_AREA)
        sm = (sm > 0).astype(np.uint8) * 255
        pooled[tile] = cv2.resize(sm, (W, H), interpolation=cv2.INTER_NEAREST)

    result_colored = np.zeros((H, W, 3), dtype=np.uint8)
    result_bw = np.zeros((H, W, 3), dtype=np.uint8)

    fovea_mask = dist <= fovea_radius
    fovea_events = fovea_mask & (on > 0)
    result_colored[fovea_events] = colors[0]
    result_bw[fovea_events] = (255, 255, 255)

    for i in range(n_rings):
        ring_mask = (dist > ring_edges[i]) & (dist <= ring_edges[i + 1])
        ring_events = ring_mask & (pooled[tiles[i]] > 0)
        result_colored[ring_events] = colors[i + 1]
        result_bw[ring_events] = (255, 255, 255)

    outer_mask = dist > ring_edges[-1]
    outer_events = outer_mask & (pooled[max_tile] > 0)
    result_colored[outer_events] = colors[min(n_rings, len(colors) - 1)]
    result_bw[outer_events] = (255, 255, 255)

    return result_bw, result_colored

def dir_color_bgr(vxv, vyv):
    # hue by direction
    ang = np.arctan2(vyv, vxv)                 # [-pi, pi]
    hue = int(((ang + np.pi) / (2*np.pi)) * 179)  # [0,179]
    hsv = np.uint8([[[hue, 255, 255]]])        # full sat/value
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
    return bgr

def run_SEMD(semd, events, HEIGHT, WIDTH):
    xs = events.x
    ys = events.y
    ts = events.ts

    S = min(HEIGHT, WIDTH)
    y0 = (HEIGHT - S) // 2
    x0 = (WIDTH - S) // 2

    mask = (xs >= x0) & (xs < x0 + S) & (ys >= y0) & (ys < y0 + S)
    xs_roi = xs[mask] - x0
    ys_roi = ys[mask] - y0
    ts_roi = ts[mask]

    # Build sourceGenn in ROI coordinates
    sourceGenn = SEMDRunner.events_to_sourceGenn(xs_roi, ys_roi, ts_roi, S, S)

    # Sim steps must match the same time base used in sourceGenn (ts_roi)
    if len(ts_roi) == 0:
        sim_steps = 1
    else:
        t0 = ts_roi.min()
        t1 = ts_roi.max()
        sim_steps = int(np.ceil((t1 - t0) * 1e-3)) + 1
        sim_steps = max(sim_steps, 1)

    TB, BT, LR, RL = semd.run_window(sourceGenn, sim_steps) # top->bottom, bottom->top, left->right and right->left activations, values live on edges not in pixel centers
    directions, angles, mag, conf, valid_positions = semd.decode_motion_vectors(TB,BT, LR, RL, pool_ksize = 3)
    

    # --- build visual ---
    vx, vy = directions
    motion_full = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    nz = conf > 1e-6
    if np.any(nz):
        s_nonzero = conf[nz]
        denom = float(np.percentile(s_nonzero, 95))
        if denom <= 1e-9:
            denom = float(s_nonzero.max())

        # heatmap in ROI
        sp = np.clip(conf / (denom + 1e-9), 0.0, 1.0)
        sp_inv = 1.0 - sp
        heat_roi = cv2.applyColorMap((sp_inv * 255).astype(np.uint8), cv2.COLORMAP_BONE)


        # --- inside your nz block after you have vx, vy, speed ---
        s_nonzero = conf[nz]
        denom = float(np.percentile(s_nonzero, 95))
        denom = max(denom, float(s_nonzero.max()), 1e-9)

        step = 4
        min_speed = float(np.percentile(s_nonzero, 30)) * 0.05
        min_speed = max(min_speed, 1e-6)

        Lmin, Lmax = 3.0, 25.0  # pixels in ROI

        for yy in range(0, S, step):
            for xx in range(0, S, step):
                s = float(conf[yy, xx])
                if s < min_speed:
                    continue

                vxv = float(vx[yy, xx])
                vyv = float(vy[yy, xx])
                n = (vxv*vxv + vyv*vyv) ** 0.5
                if n < 1e-9:
                    continue

                # length from magnitude (clipped + optional gamma)
                u = np.clip(s / denom, 0.0, 1.0)
                u = u ** 0.7                      # gamma < 1 boosts low speeds
                L = Lmin + u * (Lmax - Lmin)

                dx = vxv / n
                dy = vyv / n
                x2 = int(round(xx + dx * L))
                y2 = int(round(yy + dy * L))

                x2 = 0 if x2 < 0 else (S-1 if x2 >= S else x2)
                y2 = 0 if y2 < 0 else (S-1 if y2 >= S else y2)
                if x2 == xx and y2 == yy:
                    continue
                
                col = dir_color_bgr(vxv, vyv)
                cv2.arrowedLine(heat_roi, (xx, yy), (x2, y2),
                                col, 2, tipLength=0.35, line_type=cv2.LINE_AA)
                
        # place ROI into full frame
        motion_full[y0:y0 + S, x0:x0 + S] = heat_roi

    return directions, angles, mag, conf, valid_positions, motion_full 

def velocity_magnitude_heatmap(
        vx, vy,
        block=6,
        clip_pctl=95.0,
        gamma=0.7,
        min_nonzero=1e-6,
        bg_white=True,
        cmap=cv2.COLORMAP_TURBO,
        arrow_len_px=80,          # length in full-res pixels
        arrow_thickness=3,
        arrow_color=(0, 0, 0),    # BGR (black)
        arrow_offset_px=(120, 0), # (dx, dy) from image center: places arrow "next to" object
        weighted_global=True,     # weight mean direction by speed
        divider_axis_img=None,   # (ax,ay) unit in image coords; if provided, draw divider
        away_side_sign=+1,          # choose which half is "away"
        away_gray=(180, 180, 180),  # BGR
        divider_center_xy=None,     # (cx, cy) in pixels; if None uses image center
    ):
    H, W = vx.shape
    vx = vx.astype(np.float32)
    vy = vy.astype(np.float32)

    Hb = (H // block) * block
    Wb = (W // block) * block
    vx_c = vx[:Hb, :Wb]
    vy_c = vy[:Hb, :Wb]
    speed_c = np.sqrt(vx_c*vx_c + vy_c*vy_c)

    vx_b = vx_c.reshape(Hb//block, block, Wb//block, block).mean(axis=(1,3))
    vy_b = vy_c.reshape(Hb//block, block, Wb//block, block).mean(axis=(1,3))
    s_b  = speed_c.reshape(Hb//block, block, Wb//block, block).mean(axis=(1,3))

    nz = s_b > min_nonzero
    if not np.any(nz):
        out = np.full((H, W, 3), 255, np.uint8) if bg_white else np.zeros((H, W, 3), np.uint8)
        return out

    # --- global average motion vector over blocks (vector mean) ---
    if weighted_global:
        w = s_b[nz]                      # weight by block speed (suppresses noisy slow blocks)
        wsum = float(np.sum(w)) + 1e-12
        gvx = float(np.sum(vx_b[nz] * w) / wsum)
        gvy = float(np.sum(vy_b[nz] * w) / wsum)
    else:
        gvx = float(np.mean(vx_b[nz]))
        gvy = float(np.mean(vy_b[nz]))

    gmag = float(np.hypot(gvx, gvy))
    if gmag > 1e-9:
        gdx = gvx / gmag
        gdy = gvy / gmag
    else:
        gdx, gdy = 1.0, 0.0   # fallback direction

    # --- magnitude heatmap ---
    denom = float(np.percentile(s_b[nz], clip_pctl))
    denom = max(denom, float(s_b[nz].max()), 1e-9)
    u = np.clip(s_b / denom, 0.0, 1.0) ** float(gamma)
    u8 = (u * 255.0).astype(np.uint8)
    heat_small = cv2.applyColorMap(u8, cmap)
    if bg_white:
        heat_small[~nz] = (255, 255, 255)

   
    # gray out one half-plane (away from observer)
    if divider_axis_img is not None:
        away_blocks = halfplane_mask_blocks(
            nz=nz, block=block, H=H, W=W,
            axis_img=divider_axis_img,
            center_xy=divider_center_xy,
            side_sign=away_side_sign
        )
        heat_small[away_blocks] = away_gray

    # --- upscale ---
    heat_big = cv2.resize(heat_small, (Wb, Hb), interpolation=cv2.INTER_NEAREST)
    if Hb != H or Wb != W:
        out = np.full((H, W, 3), 255, np.uint8) if bg_white else np.zeros((H, W, 3), np.uint8)
        out[:Hb, :Wb] = heat_big
        heat_big = out

    # --- draw arrow for global mean direction (length in pixels) ---
    cx = W // 2
    cy = H // 2
    base = (int(cx + arrow_offset_px[0]), int(cy + arrow_offset_px[1]))
    tip  = (int(base[0] + arrow_len_px * gdx), int(base[1] + arrow_len_px * gdy))
    cv2.arrowedLine(heat_big, base, tip, arrow_color, arrow_thickness, tipLength=0.25)
    if divider_axis_img is not None:
        heat_big = draw_divider_line(heat_big, divider_axis_img, color=(0,0,0), thickness=2)
    # return global mean vector components (not unit)
    return heat_big

def add_orientation_wheel_legend(
        img_bgr,
        radius=55,
        thickness=10,           # ring thickness (px)
        corner="tr",            # "tr","tl","br","bl"
        pad_x=10, pad_y=10,
        inner_white=True,
        arrow_deg=None,         # optional arrow, 0=right, 90=down (image coords)
        arrow_color=(120,120,120),
        arrow_thickness=2,
    ):
    H, W = img_bgr.shape[:2]

    if corner == "tr":
        cx, cy = W - pad_x - radius - 1, pad_y + radius
    elif corner == "tl":
        cx, cy = pad_x + radius, pad_y + radius
    elif corner == "br":
        cx, cy = W - pad_x - radius - 1, H - pad_y - radius - 1
    else:  # "bl"
        cx, cy = pad_x + radius, H - pad_y - radius - 1

    x0, y0 = cx - radius, cy - radius
    x1, y1 = cx + radius + 1, cy + radius + 1
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return img_bgr

    # patch coordinates
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    rr = np.sqrt(xx*xx + yy*yy)

    outer = rr <= radius
    inner = rr < (radius - thickness)
    ring = outer & (~inner)

    # angle in image coords: 0=right, 90=down
    ang = (np.degrees(np.arctan2(yy, xx)) + 360.0) % 360.0
    hue = (ang / 2.0).astype(np.uint8)  # 0..179

    hsv = np.zeros((2*radius+1, 2*radius+1, 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = 255
    hsv[..., 2] = 255
    col = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    patch = np.zeros_like(col)
    patch[ring] = col[ring]

    if inner_white:
        patch[inner] = (255, 255, 255)

    # paste with alpha on outer disk so outside stays unchanged
    roi = img_bgr[y0:y1, x0:x1].copy()
    alpha = outer.astype(np.float32)[..., None]
    out_roi = (roi.astype(np.float32) * (1 - alpha) + patch.astype(np.float32) * alpha).astype(np.uint8)
    img_bgr[y0:y1, x0:x1] = out_roi

    # optional arrow
    if arrow_deg is not None:
        a = np.deg2rad(arrow_deg)
        r2 = radius - thickness - 2
        ex = int(round(cx + np.cos(a) * r2))
        ey = int(round(cy + np.sin(a) * r2))
        cv2.arrowedLine(img_bgr, (cx, cy), (ex, ey),
                        arrow_color, arrow_thickness,
                        tipLength=0.25, line_type=cv2.LINE_AA)

    return img_bgr

def velocity_orientation_heatmap(
        angles_deg,              # HxW, 0..360
        mask=None,               # HxW bool (e.g., active events)
        block=6,
        bg_white=True,
        sat=255,
        val=255,
        min_valid_frac=0.1,      # require some valid pixels in block
        divider_axis_img=None,   # (ax,ay) unit in image coords; if provided, draw divider
        show_axis = True
    ):
    H, W = angles_deg.shape
    Hb = (H // block) * block
    Wb = (W // block) * block

    A = angles_deg[:Hb, :Wb].astype(np.float32)

    # pixel-valid: mask AND angle!=0  (so 360 stays valid)
    if mask is None:
        pix_valid = (A != 0.0)
    else:
        M = mask[:Hb, :Wb].astype(bool)
        pix_valid = M & (A != 0.0)

    th = np.deg2rad(A)
    c = np.where(pix_valid, np.cos(th), 0.0)
    s = np.where(pix_valid, np.sin(th), 0.0)

    c_b = c.reshape(Hb//block, block, Wb//block, block).sum(axis=(1,3))
    s_b = s.reshape(Hb//block, block, Wb//block, block).sum(axis=(1,3))
    cnt = pix_valid.reshape(Hb//block, block, Wb//block, block).sum(axis=(1,3)).astype(np.float32)

    # block-valid: enough valid pixels in the block
    valid_b = cnt >= (min_valid_frac * block * block)

    ang_b = (np.rad2deg(np.arctan2(s_b, c_b)) + 360.0) % 360.0

    hue = (ang_b / 2.0).astype(np.uint8)  # 0..179
    S = np.full_like(hue, sat, dtype=np.uint8)
    V = np.full_like(hue, val, dtype=np.uint8)

    hsv = np.stack([hue, S, V], axis=-1)
    bgr_small = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # force invalid blocks to white/black (independent of hue)
    if bg_white:
        bgr_small[~valid_b] = (255, 255, 255)
    else:
        bgr_small[~valid_b] = (0, 0, 0)

    bgr = cv2.resize(bgr_small, (Wb, Hb), interpolation=cv2.INTER_NEAREST)
    if Hb != H or Wb != W:
        out = np.full((H, W, 3), 255, np.uint8) if bg_white else np.zeros((H, W, 3), np.uint8)
        out[:Hb, :Wb] = bgr
        bgr = out

    if divider_axis_img is not None and show_axis:
        bgr = draw_divider_line(bgr, divider_axis_img, color=(0,0,0), thickness=1)
    bgr = add_orientation_wheel_legend(bgr, corner="tr", radius=25, pad_x=10, pad_y=50)
    return bgr

def rotation_axis_from_proprioception(model, data, cam_id=0, q_delta=None):
    # Ensure camera xmat is current
    mujoco.mj_forward(model, data)

    # 1) axis in world from q_delta = [w, x, y, z]
    w, x, y, z = q_delta
    axis_w = np.array([x, y, z], dtype=np.float64)
    n = np.linalg.norm(axis_w)
    if n < 1e-9:
        return None
    axis_w /= n

    # 2) camera basis in world (right, up)
    # Using MuJoCo camera rotation matrix (world-from-camera): data.cam_xmat[cam_id] is 3x3 row-major

    R = np.asarray(data.cam_xmat[cam_id]).reshape(3, 3)
    right_w = R[:, 0]        # image right in world
    up_w    = R[:, 1]        # image up in world
    fwd_w   = -R[:, 2]       # camera forward in world
    right_w = right_w / (np.linalg.norm(right_w) + 1e-12)
    up_w    = up_w    / (np.linalg.norm(up_w) + 1e-12)
    fwd_w   = fwd_w   / (np.linalg.norm(fwd_w) + 1e-12)

    # 3) project axis onto image plane basis
    ax = float(np.dot(axis_w, right_w))   # +x pixel is right
    ay = float(np.dot(axis_w, up_w))      # +ay is "up" in camera sense

    # convert to pixel coordinates (y down):
    ay = -ay

    n = np.hypot(ax, ay)
    if n < 1e-9:
        return None
    return (ax / n, ay / n)

def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return q / (n + 1e-12)

def quat_inv(q):
    # for unit quaternions, inverse = conjugate
    q = quat_normalize(q)
    return quat_conj(q)


def quat_to_axis_angle(q, eps=1e-12):
    """
    q: [w,x,y,z] (assumed unit or close)
    returns: angle (rad), ux, uy, uz
    """
    q = quat_normalize(q)
    # enforce shortest-arc convention
    if q[0] < 0:
        q = -q

    w, x, y, z = q
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)

    s = np.sqrt(max(0.0, 1.0 - w*w))
    if s < eps:
        # near zero rotation: axis arbitrary
        return 0.0, 1.0, 0.0, 0.0

    ux, uy, uz = x/s, y/s, z/s
    return angle, ux, uy, uz


def relative_rotation_axis_angle(q_prev, q_curr):
    """
    Relative rotation from prev->curr:
      q_rel = inv(q_prev) ⊗ q_curr
    Returns axis-angle of q_rel (rad).
    """
    q_prev = quat_normalize(q_prev)
    q_curr = quat_normalize(q_curr)
    q_rel = quat_mul(quat_inv(q_prev), q_curr)

    return quat_to_axis_angle(q_rel)

# Apply rotation adjustment each {rotate_each_n_steps} steps, otherwise continue in original direction
def next_rotation_correction_and_axis(next_fixation, cx, cy, model, data, abs_pitch, abs_roll, d_pitch, d_roll, 
                                       DEG_PER_PX, MAX_STEP_DEG, divider_axis_img=None, q_prev=None, 
                                       base_q=None, rotate_each_n_steps=1, step_index=0, random_prob=0.05):

    if step_index % rotate_each_n_steps == 0:
        if np.random.rand() < random_prob:
            # Random direction at max angular magnitude
            angle = np.random.uniform(0, 2 * np.pi)
            d_pitch = float(10 * np.sin(angle))
            d_roll  = float(10 * np.cos(angle))
        else:
            ex = next_fixation[0] - cx
            ey = next_fixation[1] - cy
            d_pitch = float(np.clip((ey * DEG_PER_PX), -MAX_STEP_DEG, MAX_STEP_DEG))
            d_roll  = float(np.clip(-(ex * DEG_PER_PX), -MAX_STEP_DEG, MAX_STEP_DEG))

        abs_pitch += d_pitch
        abs_roll  += d_roll
        q_curr = quat_normalize(get_total_quaternion(abs_pitch, abs_roll, base_q))
        if q_curr[0] < 0:
            q_curr = -q_curr
        if q_prev is None:
            divider_axis_img = None
        else:
            q_delta_obj = quat_mul(q_curr, quat_conj(q_prev))
            q_delta_obj = quat_normalize(q_delta_obj)
            if q_delta_obj[0] < 0:
                q_delta_obj = -q_delta_obj
            divider_axis_img = rotation_axis_from_proprioception(model, data, cam_id=0, q_delta=q_delta_obj)
    else:
        abs_pitch += d_pitch
        abs_roll  += d_roll
        q_curr = quat_normalize(get_total_quaternion(abs_pitch, abs_roll, base_q))
        if q_curr[0] < 0:
            q_curr = -q_curr

    return divider_axis_img, q_curr, abs_pitch, abs_roll, d_pitch, d_roll

def blockwise_circular_mean_angles(angles_deg, block=6, invalid_is_zero=True):
    H, W = angles_deg.shape
    Hb = (H // block) * block
    Wb = (W // block) * block

    A = angles_deg[:Hb, :Wb].astype(np.float32)

    if invalid_is_zero:
        pix_valid = (A != 0.0)
    else:
        pix_valid = np.ones_like(A, dtype=bool)

    th = np.deg2rad(A)
    c = np.where(pix_valid, np.cos(th), 0.0)
    s = np.where(pix_valid, np.sin(th), 0.0)

    c_b = c.reshape(Hb//block, block, Wb//block, block).sum(axis=(1,3))
    s_b = s.reshape(Hb//block, block, Wb//block, block).sum(axis=(1,3))
    cnt = pix_valid.reshape(Hb//block, block, Wb//block, block).sum(axis=(1,3)).astype(np.float32)

    valid_b = cnt > 0
    ang_b = (np.rad2deg(np.arctan2(s_b, c_b)) + 360.0) % 360.0  # [0,360)

    # set invalid blocks to 0 sentinel (so later you can whiten them)
    ang_b2 = ang_b.copy()
    ang_b2[~valid_b] = 0.0

    # upsample to cropped size, then pad to full size
    ang_crop = cv2.resize(ang_b2.astype(np.float32), (Wb, Hb), interpolation=cv2.INTER_NEAREST)
    valid_crop = cv2.resize(valid_b.astype(np.uint8)*255, (Wb, Hb), interpolation=cv2.INTER_NEAREST) > 0

    ang_full = np.zeros((H, W), dtype=np.float32)
    valid_full = np.zeros((H, W), dtype=bool)
    ang_full[:Hb, :Wb] = ang_crop
    valid_full[:Hb, :Wb] = valid_crop

    return ang_full, valid_full

def angle_cosine_similarity_heatmap(
            angles_meas_deg,          # HxW, deg (0 invalid, 360 valid allowed)
            angles_exp_deg,           # HxW, deg (0 invalid, 360 valid allowed)
            block=12,
            min_valid_frac=1.0,
            clip_pctl=99.0,
            gamma=1.0,
            bg_white=True,
            cmap=cv2.COLORMAP_TURBO,
        ):
            """
            Blockwise cosine similarity of angle difference.
            Returns:
            heat_bgr: HxW×3 uint8 heatmap (blockwise upsampled)
            cos_b_up: HxW float32 blockwise cos similarity upsampled (invalid blocks=0)
            valid_up: HxW bool blockwise validity upsampled
            """
            H, W = angles_meas_deg.shape
            Hb = (H // block) * block
            Wb = (W // block) * block

            A = (angles_meas_deg[:Hb, :Wb].astype(np.float32) % 360.0)
            B = (angles_exp_deg[:Hb, :Wb].astype(np.float32) % 360.0)

            pix_valid = (angles_meas_deg[:Hb, :Wb] != 0.0) & (angles_exp_deg[:Hb, :Wb] != 0.0)

            # circular diff in [-180,180]
            d = np.zeros((Hb, Wb), dtype=np.float32)
            d[pix_valid] = (A[pix_valid] - B[pix_valid] + 180.0) % 360.0 - 180.0

            cos_pix = np.zeros((Hb, Wb), dtype=np.float32)
            cos_pix[pix_valid] = np.cos(np.deg2rad(d[pix_valid]))  # [-1,1]

            # blockwise mean cosine similarity
            sum_b = cos_pix.reshape(Hb//block, block, Wb//block, block).sum(axis=(1,3))
            cnt_b = pix_valid.reshape(Hb//block, block, Wb//block, block).sum(axis=(1,3)).astype(np.float32)
            cos_b = sum_b / (cnt_b + 1e-6)

            valid_b = cnt_b > 0.0

            # set invalid blocks to 0
            cos_b2 = cos_b.copy()
            cos_b2[~valid_b] = 0.0

            # upsample to cropped size then pad
            cos_up = cv2.resize(cos_b2.astype(np.float32), (Wb, Hb), interpolation=cv2.INTER_NEAREST)
            valid_up = cv2.resize((valid_b.astype(np.uint8)*255), (Wb, Hb), interpolation=cv2.INTER_NEAREST) > 0

            cos_full = np.zeros((H, W), dtype=np.float32)
            valid_full = np.zeros((H, W), dtype=bool)
            cos_full[:Hb, :Wb] = cos_up
            valid_full[:Hb, :Wb] = valid_up

            # map [-1,1] -> [0,255]
            u = (cos_full + 1.0) * 0.5

            if np.any(valid_full):
                lo = np.percentile(u[valid_full], 100.0 - clip_pctl) if clip_pctl < 100 else 0.0
                hi = np.percentile(u[valid_full], clip_pctl)
                if hi <= lo:
                    lo, hi = 0.0, 1.0
                u = np.clip((u - lo) / (hi - lo + 1e-12), 0.0, 1.0)

            u = u ** float(gamma)
            u8 = (u * 255.0).astype(np.uint8)

            heat_bgr = cv2.applyColorMap(u8, cmap)
            if bg_white:
                heat_bgr[~valid_full] = (255,255,255)
            else:
                heat_bgr[~valid_full] = (0,0,0)

            return heat_bgr, cos_full, valid_full

def pad_center(arr, target_h, target_w, pad_value=0):
            H, W = arr.shape[:2]
            pad_h = target_h - H
            pad_w = target_w - W
            if pad_h < 0 or pad_w < 0:
                raise ValueError("target smaller than input")

            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            if arr.ndim == 2:
                return np.pad(arr, ((top, bottom), (left, right)),
                            mode="constant", constant_values=pad_value)
            else:
                return np.pad(arr, ((top, bottom), (left, right), (0, 0)),
                            mode="constant", constant_values=pad_value)


def infere_angles_from_proprioception(ev_img, divider_axis_img ):
    HEIGHT = ev_img.shape[0]
    WIDTH = ev_img.shape[1]
    expected_angles_proprioception = np.zeros([HEIGHT, WIDTH])
    mask = (ev_img[..., 0] > 0) | (ev_img[..., 1] > 0) | (ev_img[..., 2] > 0)
    
    if divider_axis_img is not None:
        ax, ay =  divider_axis_img  
    else:
        ax,ay = 0,0

    # axis angle in image coords (0°=right, 90°=down), in [0,360)
    theta_axis = (np.degrees(np.arctan2(ay, ax)) + 360.0) % 360.0

    # perpendicular directions (two opposite normals), in [0,360)
    theta_perp_1 = (theta_axis + 90.0) % 360.0
    theta_perp_2 = (theta_axis - 90.0) % 360.0

    px, py = ay, -ax   # +90° normal

    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    cx, cy = (WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0
    side = (xx - cx) * px + (yy - cy) * py   
    
    out = np.where(side >= 0, theta_perp_1, theta_perp_2).astype(np.float32)
    expected_angles_proprioception[mask] = out[mask]
    return expected_angles_proprioception

def axis_sloped_weight(
    H, W,
    divider_axis_img,          # (ax, ay) along divider line
    center_xy=None,            # (cx, cy) where divider passes
    towards_side_sign=+1,      # choose which half-plane is "towards"
    boost=3.0,                 # max multiplicative boost on towards side at far edge
    suppress=0.3,              # min multiplicative weight on away side at far edge
    gamma=1.0,                 # nonlinearity on distance ramp
    clip_min=0.0, clip_max=1.0 # clamp for normalized distance
):
    ax, ay = divider_axis_img
    n = np.hypot(ax, ay) + 1e-12
    ax, ay = ax / n, ay / n

    # normal to divider line (this splits the half-planes)
    px, py = -ay, ax

    if center_xy is None:
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    else:
        cx, cy = center_xy

    yy, xx = np.mgrid[0:H, 0:W]
    side = (xx - cx) * px + (yy - cy) * py            # signed distance (up to scale)
    side *= float(towards_side_sign)                  # positive = towards side

    # normalize by maximum absolute distance in the image (so 0..1 scale)
    maxd = float(np.max(np.abs(side))) + 1e-12
    d = np.abs(side) / maxd                            # 0..1 distance from axis line
    d = np.clip(d, clip_min, clip_max) ** float(gamma)

    # build weights: ramp up on towards side, ramp down on away side
    # towards: 1 -> boost, away: 1 -> suppress
    w = np.ones((H, W), dtype=np.float32)
    toward = side >= 0
    away = ~toward

    w[toward] = 1.0 + (float(boost) - 1.0) * d[toward]
    w[away]   = 1.0 - (1.0 - float(suppress)) * d[away]

    return w

def modulate_saliency_with_surprise(
    saliency,          # HxW float (raw saliency, not colormapped)
    surprise,          # HxW float (e.g., cos_sim in [-1,1] or surprise_deg in [0,180])
    valid=None,        # HxW bool mask; if None use finite check
    mode="cos_sim",    # "cos_sim" or "deg"
    alpha=1.0,         # modulation strength
    eps=1e-9,
):
    """
    Returns saliency_mod: HxW float, same scale as saliency.
    Most surprise => larger weight.
    For cos_sim: surprise = 1-match, -1-opposite => weight uses (1 - cos_sim)/2.
    For deg: surprise in [0,180] => normalized by 180.
    """
    
    S = saliency.astype(np.float32).copy()

    if valid is None:
        valid = np.isfinite(surprise)

    if mode == "cos_sim":
        # convert similarity to surprise in [0,1]
        # cos_sim=1 -> 0 surprise, cos_sim=-1 -> 1 surprise
        U = np.zeros_like(S, dtype=np.float32)
        U[valid] = 0.5 * (1.0 - surprise[valid].astype(np.float32))
    elif mode == "deg":
        U = np.zeros_like(S, dtype=np.float32)
        U[valid] = np.clip(surprise[valid].astype(np.float32) / 180.0, 0.0, 1.0)
    else:
        raise ValueError("mode must be 'cos_sim' or 'deg'")

    # weight: 1 + alpha*U  (alpha=1 -> up to 2x boost)
    W = 1.0 + float(alpha) * U

    S_mod = S * W

    # optional: renormalize to keep comparable range (comment out if you want raw scaling)
    m = S_mod.max()
    if m > eps: S_mod = S_mod / m * S.max()
    
    return S_mod, W

def downsample_angles_blocks_fill(
        angles_deg,          # HxW, 0 invalid, 360 valid=0/360
        block=12,
        out_shape=None,      # e.g. (346,260) to force back; default = input shape
    ):
        H, W = angles_deg.shape
        if out_shape is None:
            out_shape = (H, W)

        Hb = (H // block) * block
        Wb = (W // block) * block
        A = angles_deg[:Hb, :Wb].astype(np.float32)

        # valid pixels: anything nonzero
        valid = (A != 0.0)

        # map 360 -> 0 for trig
        A0 = (A % 360.0)

        th = np.deg2rad(A0)
        c = np.where(valid, np.cos(th), 0.0)
        s = np.where(valid, np.sin(th), 0.0)

        # circular sum per block
        c_b = c.reshape(Hb//block, block, Wb//block, block).sum(axis=(1,3))
        s_b = s.reshape(Hb//block, block, Wb//block, block).sum(axis=(1,3))
        cnt = valid.reshape(Hb//block, block, Wb//block, block).sum(axis=(1,3))

        valid_b = cnt > 0

        ang_b = (np.rad2deg(np.arctan2(s_b, c_b)) + 360.0) % 360.0  # [0,360)
        # encode 0-direction as 360 (to match your convention) ONLY for valid blocks
        ang_b = ang_b.astype(np.float32)
        ang_b[valid_b & (ang_b == 0.0)] = 360.0
        ang_b[~valid_b] = 0.0
        return ang_b 