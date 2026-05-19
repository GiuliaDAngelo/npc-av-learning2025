
import numpy as np
import cv2
from PIL import Image
from IPython.display import display, clear_output
from time import sleep
global HEIGHT
global WIDTH

HEIGHT = 260
WIDTH = 346

def make_colorbar(height, width=50, cmap=cv2.COLORMAP_JET):
    gradient = np.linspace(255,0, height).astype(np.uint8)
    gradient = np.repeat(gradient[:, None], width, axis=1)
    colorbar = cv2.applyColorMap(gradient, cmap)
    return colorbar


def put_title(img, title, y=24, pad=6, font_scale=0.6, thickness=1,
              text_color=(0,0,0), bg_color=(255,255,255)):
    """
    img: HxWx3 BGR
    draws a small solid header band and title text at the top-left
    """
    out = img.copy()
    h, w = out.shape[:2]
    band_h = int(y + pad)
    band_h = min(band_h, h)

    # header band
    cv2.rectangle(out, (0, 0), (w, band_h), bg_color, -1)

    # text
    cv2.putText(out, title, (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                text_color, thickness, cv2.LINE_AA)
    return out

def show_animation(rgb_img, event_img, saliency_map, inhibited_saliency_map, foveated_img, fov_radius,
                   x, y, motion_img, expected_motion, surprise, modulated_saliency, averaged_semd, next_look, H, W,pause_s=0.03):

    cv2.drawMarker(saliency_map, (x, y),
                   color=(255, 255, 255),
                   markerType=cv2.MARKER_CROSS,
                   markerSize=25, thickness=2)
    
    cv2.drawMarker(modulated_saliency, (next_look[0], next_look[1]),
                   color=(255, 255, 255),
                   markerType=cv2.MARKER_CROSS,
                   markerSize=25, thickness=2)
    
    cv2.drawMarker(inhibited_saliency_map, (next_look[0], next_look[1]),
                   color=(255, 255, 255),
                   markerType=cv2.MARKER_CROSS,
                   markerSize=25, thickness=2)
    
    # Ensure consistent tile sizes
    #rgb_img       = cv2.resize(rgb_img,       (W, H))
    #event_img     = cv2.resize(event_img,     (W, H))
    #saliency_map  = cv2.resize(saliency_map,  (W, H))
    #foveated_img  = cv2.resize(foveated_img,  (W, H))


    # pad motion_img to 640x480 from square
    H, W, C = motion_img.shape
    H_new, W_new =  event_img.shape[0],event_img.shape[1]

    pad_h = H_new - H   # 160
    pad_w = W_new - W   # 0

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    motion_img_2 = np.pad(
       motion_img,
        pad_width=((top, bottom), (left, right), (0, 0)),
        mode="constant",
        constant_values=255
    )


    cv2.circle(foveated_img, (next_look[0], next_look[1]), fov_radius,
               (255,255,0), 2)

    colorbar = make_colorbar(height=H_new)

    empty = np.ones((H, W, 3), dtype=np.uint8) * 255
    empty_color = np.ones_like(colorbar)

    rgb_img       = put_title(rgb_img, "RGB 3D object")
    saliency_map  = put_title(saliency_map, "Saliency map from visual attention")
    event_img     = put_title(event_img, "Simulated Events")
    motion_img_2  = put_title(motion_img_2, "SEMD directions from events")
    expected_motion = put_title(expected_motion, "Expected directions from proprioception")
    surprise      = put_title(surprise, "Cos similarity of expected vs actual directionality")
    modulated_saliency = put_title(modulated_saliency, "SEMD + proprioception modulated saliency")
    averaged_semd = put_title(averaged_semd, "Average direction blocks SEMD")

    # Now both rows have EXACTLY 5 tiles of identical size
    top_row = np.hstack([rgb_img,       motion_img_2,255*empty_color,       saliency_map, colorbar])

    colorbar = make_colorbar(height=H,cmap=cv2.COLORMAP_TURBO)
    middle_row = np.hstack([event_img,  expected_motion, 255*empty_color, inhibited_saliency_map, colorbar ])
    third_row = np.hstack([ empty,      surprise,        colorbar,        modulated_saliency,     colorbar ])
    bottom_row = np.hstack([averaged_semd,      empty, 255*empty_color,       foveated_img,           empty_color])


    grid = np.vstack([top_row, middle_row,third_row, bottom_row])

    vis = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
    clear_output(wait=True)
    display(Image.fromarray(vis))
    sleep(pause_s)

