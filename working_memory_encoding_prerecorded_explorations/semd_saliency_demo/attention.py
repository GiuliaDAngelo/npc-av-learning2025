import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import sinabs.layers as sl
from scipy.special import iv

from skimage.transform import rescale, resize, downscale_local_mean

import cv2 as cv
import torch
import torchvision


class Config:
    # Constants
    MAX_X, MAX_Y = 480, 640 # ADJUST FOR DESIRED RESOLUTION!!!
    RESOLUTION = (MAX_Y, MAX_X)
    # CAMERA_EVENTS = 'right'
    # CODEC = '24bit'

    # Attention Parameters
    ATTENTION_PARAMS = {
        'size_krn' : 32, # Size of the kernel
        'r0' : 5 , # Radius shift from the center 15
        'rho' : 0.5 , # Scale coefficient to control arc length
        'theta' : np.pi * 1/8, # Angle to control the orientation of the arc
        'thetas' : np.arange(0, 2 * np.pi, np.pi / 4),
        'thick' : 20, # thickness of the arc
        'offsetpxs' : 16, # size / 2
        'offset' : (1,1),
        'fltr_resize_perc' : [3,3],
        'num_pyr' : 6,
        'tau_mem': 33, # in units of simulation steps!!
        'stride':1,
        'out_ch':1
    }

    salmax_coords = np.zeros((2,), dtype=np.int32)
    # DATA_PARAMS = {
    #     'TIME_WINDOW': 12000, #10 ms #1 ms is 100 steps
    #     'SCAN_PATH': 12000 #120 ms
    # }





def zero_2pi_tan(x, y):
    """
    Compute the angle in radians between the positive x-axis and the point (x, y),
    ensuring the angle is in the range [0, 2π].

    Args:
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.

    Returns:
        angle (float): Angle in radians, between 0 and 2π.
    """
    angle = np.arctan2(y, x) % (2 * np.pi)  # Get the angle in radians and wrap it in the range [0, 2π]
    return angle


def vm_filter(theta, scale, rho=0.1, r0=0, thick=0.5, offset=(0, 0)):
    """Generate a Von Mises filter with r0 shifting and an offset."""
    height, width = scale, scale
    vm = np.empty((height, width))
    offset_x, offset_y = offset

    for x in range(width):
        for y in range(height):
            # Shift X and Y based on r0 and offset
            X = (x - width / 2) + r0 * np.cos(theta) - offset_x * np.cos(theta)
            Y = (height / 2 - y) + r0 * np.sin(theta) - offset_y * np.sin(theta)  # Inverted Y for correct orientation
            r = np.sqrt(X**2 + Y**2)
            angle = zero_2pi_tan(X, Y)

            # Compute the Von Mises filter value
            vm[y, x] = np.exp(thick*rho * r0 * np.cos(angle - theta)) / iv(0, r - r0)
    # normalise value between -1 and 1
    # vm = vm / np.max(vm)
    # vm = vm * 2 - 1
    return vm

def VMkernels(thetas, size, rho, r0, thick, offset,fltr_resize_perc):
    """
    Create a set of Von Mises filters with different orientations.

    Args:
        thetas (np.ndarray): Array of angles in radians.
        size (int): Size of the filter.
        rho (float): Scale coefficient to control arc length.
        r0 (int): Radius shift from the center.

    Returns:
        filters (list): List of Von Mises filters.
    """
    filters = []
    for theta in thetas:
        filter = vm_filter(theta, size, rho=rho, r0=r0, thick=thick, offset=offset)
        filter = rescale(filter, fltr_resize_perc, anti_aliasing=False)
        filters.append(filter)
    filters = torch.tensor(np.stack(filters).astype(np.float32))
    if __name__ == '__main__':
        plot_filters(filters, thetas)
    return filters

def plot_filters(filters, angles):
    """
    Plot the von Mises filters using matplotlib.

    Args:
        filters (torch.Tensor): A tensor containing filters to be visualized.
    """
    # Create subplots for 8 orientation VM filters
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle(f'VM filters size ({filters.shape[1]},{filters.shape[2]})', fontsize=16)
    fig.suptitle(f'VM filters size ({filters.shape[1]},{filters.shape[2]})', fontsize=16)

    # Display filters with their corresponding angles
    for i in range(8):
        if i < 4:
            axes[0, i].set_title(f"{round(angles[i],2)} grad")
            axes[0, i].imshow(filters[i])
            plt.colorbar(axes[0, i].imshow(filters[i]))
        else:
            axes[1, i - 4].set_title(f"{round(angles[i],2)} grad")
            axes[1, i - 4].imshow(filters[i])
            plt.colorbar(axes[1, i - 4].imshow(filters[i]))
    # add color bar to see the values of the filters
    plt.show()

def plot_kernel(kernel,size):
    #plot kernel 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = torch.linspace(-size // 2, size // 2, size)
    y = torch.linspace(-size // 2, size // 2, size)
    x, y = torch.meshgrid(x, y, indexing='ij')
    ax.plot_surface(x.numpy(), y.numpy(), kernel.numpy(), cmap='jet')
    plt.show()


def net_def(filter, tau_mem, in_ch, out_ch, size_krn, device, stride):
    # define our single layer network and load the filters
    net = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, (size_krn,size_krn), stride=stride, bias=False), # padding='same'
        sl.LIF(tau_mem),
    )
    net[0].weight.data = filter.unsqueeze(1).to(device)
    net[1].v_mem = net[1].tau_mem * net[1].v_mem.to(device)
    return net


def initialise_attention(device, ATTENTION_PARAMS):
    vm_kernels = VMkernels(
        ATTENTION_PARAMS['thetas'], ATTENTION_PARAMS['size_krn'],
        ATTENTION_PARAMS['rho'], ATTENTION_PARAMS['r0'], ATTENTION_PARAMS['thick'],
        ATTENTION_PARAMS['offset'], ATTENTION_PARAMS['fltr_resize_perc']
    )
    net_attention = net_def(vm_kernels, ATTENTION_PARAMS['tau_mem'], ATTENTION_PARAMS['num_pyr'], ATTENTION_PARAMS['out_ch'],
                         ATTENTION_PARAMS['size_krn'], device, ATTENTION_PARAMS['stride'])

    return net_attention

def run_attention(window, net, device, resolution, size_krn_after_oms, num_pyr):
    with torch.no_grad():
        # Create resized versions of the frames
        resized_frames = [
            torchvision.transforms.Resize((int(resolution[0] / pyr), int(resolution[1] / pyr)), antialias=False)(
                window.cpu()) for pyr in range(1, num_pyr + 1)]
        # Process frames in batches
        batch_frames = torch.stack(
            [torchvision.transforms.Resize((resolution[0], resolution[1]))(window) for window in resized_frames]).type(torch.float32)
        batch_frames = batch_frames.to(device)  # Move to GPU if available
        output_rot = net(batch_frames)

        # Sum the outputs over rotations and scales
        output_rot_sum = torch.sum(torch.sum(output_rot, dim=1, keepdim=True), dim=0, keepdim=True).type(torch.float32).cpu().detach()
        if isinstance(size_krn_after_oms, (tuple, list)):
            Ht, Wt = int(size_krn_after_oms[0]), int(size_krn_after_oms[1])
        else:
            Ht = Wt = int(size_krn_after_oms)

        salmap_t = torch.nn.functional.interpolate(
            output_rot_sum, size=(Ht, Wt), mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)

        salmap = salmap_t.numpy()
        den = (salmap.max() - salmap.min())
        if den <= 1e-12:
            salmap_n = np.zeros_like(salmap, dtype=np.float32)
        else:
            salmap_n = (salmap - salmap.min()) / den
        salmap_u8 = salmap_n * 255.0

        salmax_coords = np.unravel_index(np.argmax(salmap_u8), salmap_u8.shape)



    return salmap,salmax_coords



def visualize_saliency(frame, salmap, salmax_coords):
    """
    Overlay saliency map and highlight the most salient point.

    Args:
        frame (np.ndarray): Original event frame, HxW or HxWx3.
        salmap (np.ndarray): Saliency map, HxW, values 0-255.
        salmax_coords (tuple): (y, x) coordinates of most salient point.

    Returns:
        np.ndarray: Saliency map overlaid with highlighted point (color image).
    """
    salmap_uint8 = salmap.astype(np.uint8)
    salmap_color = cv.applyColorMap(salmap_uint8, cv.COLORMAP_JET)
    cv.circle(salmap_color, (salmax_coords[1], salmax_coords[0]), radius=5, color=(255,255,255), thickness=-1)

    h, w, _ = salmap_color.shape
    legend_w = 40
    legend = np.zeros((h, legend_w, 3), dtype=np.uint8)

    # Create a vertical gradient (0-255) for legend
    for i in range(h):
        value = int(255 * (1 - i / h))  # top = max, bottom = min
        legend[i, :] = (value, value, value)

    # Apply same colormap
    legend_color = cv.applyColorMap(legend[:, :, 0], cv.COLORMAP_JET)

    # Combine image and legend
    salmap_color = np.hstack((salmap_color, legend_color))
    # Put min/max labels
    cv.putText(salmap_color, "High", (w + 5, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv.putText(salmap_color, "Low", (w + 5, h - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return salmap_color



def rescale_coords(x_sal, y_sal, sal_dim, frame_dim):
    # version 1
    # x_frame = x_sal * (260 / 241)
    # y_frame = y_sal * (346 / 327)

    # version 2
    x_frame = x_sal
    y_frame = y_sal

    # version 3
    # sal_map_centre_x, sal_map_centre_y, _ = sal_dim
    # sal_map_centre_x = int(sal_map_centre_x // 2)
    # sal_map_centre_y = int(sal_map_centre_y // 2)
    # frame_centre = [v // 2 for v in frame_dim]
    # x_frame = (x_sal - sal_map_centre_x) * (frame_dim[0] / sal_dim[0]) + frame_centre[0]
    # y_frame = (y_sal - sal_map_centre_y) * (frame_dim[1] / sal_dim[1]) + frame_centre[1]

    return int(x_frame), int(y_frame)


def visualize_isolate_roi(frame, center, size=20):
    """
    Keep only a square region of interest (ROI) around a center point.
    Everything else becomes black.
    """
    h, w = frame.shape[:2] # h = 260, w = 346
    # x = height, y = w
    x, y = int(center[0]), int(center[1])
    half = size // 2

    # ROI boundaries, clamped to image limits
    x1, x2 = max(0, y - half), min(w, y + half)
    y1, y2 = max(0, x - half), min(h, x + half)

    # Create a black canvas
    masked = np.zeros_like(frame)

    # Copy only ROI pixels
    masked[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    if len(frame.shape) == 2:
        masked = cv.cvtColor(masked, cv.COLOR_GRAY2BGR)
    cv.rectangle(masked, (x1, y1), (x2, y2), (0, 255, 255), 1)
    cv.circle(masked, (y, x), radius=2, color=(255,255,0), thickness=-1)

    return masked

# Todo check if both functions return the same roi, redo visualization for function below

def isolate_roi_events(events, center, frame_shape, size=20):
    """
    Filter dv_processing.Event objects to only those inside the ROI.
    """
    h, w = frame_shape[:2]
    cx, cy = int(center[0]), int(center[1])
    half = size // 2

    # ROI bounds clamped to frame
    x_min, x_max = max(0, cx - half), min(w, cx + half)
    y_min, y_max = max(0, cy - half), min(h, cy + half)

    # Keep only events inside ROI
    filtered = [
        e for e in events
        if x_min <= e.x() < x_max and y_min <= e.y() < y_max
    ]

    return filtered, x_min, y_min

if __name__ == '__main__':
    config = Config()
    initialise_attention("cpu", config.ATTENTION_PARAMS)