import numpy as np
import cv2
import numpy as np
import cv2
from collections import deque
from scipy.special import iv
import torch
import numpy as np
from scipy.special import iv
import torch
import torch.nn as nn
import sinabs.layers as sl
from skimage.transform import rescale, resize, downscale_local_mean
import torchvision
import torch.nn.functional as F



def net_def(filter, tau_mem, in_ch, out_ch, size_krn, device, stride):
    # define our single layer network and load the filters
    net = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, (size_krn,size_krn),  stride=stride, bias=False),
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
    return filters


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


def run_attention1(window, net, device, resolution, num_pyr):
    # Create resized versions of the frames
    resized_frames = [torchvision.transforms.Resize((int(resolution[0] / num_pyr), int(resolution[1] / num_pyr)))(
        window) for pyr in range(1, num_pyr + 1)]

    # Process frames in batches
    batch_frames = torch.stack(
        [torchvision.transforms.Resize((resolution[0], resolution[1]))(window) for window in resized_frames]).type(torch.float32)
    batch_frames = batch_frames.to(device)  # Move to GPU if available
    output_rot = net(batch_frames)
    # Sum the outputs over rotations and scales
    output_rot_sum = torch.sum(torch.sum(output_rot, dim=1, keepdim=True), dim=0, keepdim=True).type(torch.float32).cpu().detach()
    salmap = torchvision.transforms.Resize((resolution[0], resolution[1]))(output_rot_sum).squeeze(0).squeeze(
        0)
    salmax_coords = np.unravel_index(torch.argmax(salmap).cpu().numpy(), salmap.shape)
    # normalise salmap for visualization
    salmap = salmap.detach().cpu().numpy()
    salmap = np.array((salmap - salmap.min()) / (salmap.max() - salmap.min()) * 255)
    return salmap,salmax_coords


def run_attention(window, net, device, resolution, num_pyr):
    """
    window:  torch tensor [1, 1, H, W] or [1, 3, H, W]
    net:     saliency network (already .to(device).eval())
    device:  torch.device("mps") or "cpu"
    resolution: (H_out, W_out)
    num_pyr: number of pyramid scales
    """

    # make sure this tensor is detached from any previous graph
    window = window.detach()

    H_out, W_out = resolution

    # we don't need gradients for attention
    with torch.no_grad():
        # build pyramid on CPU (small tensors) – stays cheaper on MPS
        resized_frames = [
            torchvision.transforms.Resize(
                (int(H_out / num_pyr), int(W_out / num_pyr))
            )(window)
            for _ in range(1, num_pyr + 1)
        ]

        # upscale all pyramid levels to the same resolution
        batch_frames = torch.stack([
            torchvision.transforms.Resize((H_out, W_out))(w)
            for w in resized_frames
        ]).to(device, dtype=torch.float32)   # [num_pyr, C, H_out, W_out]

        # forward pass on MPS/CPU
        output_rot = net(batch_frames)       # shape [num_rot, num_pyr, H_out, W_out] or similar

        # sum over rotations and scales
        output_rot_sum = output_rot.sum(dim=1, keepdim=True).sum(dim=0, keepdim=True)
        output_rot_sum = output_rot_sum.to("cpu", dtype=torch.float32)  # move off MPS

        # resize to target resolution (still on CPU)
        salmap_t = torchvision.transforms.Resize((H_out, W_out))(output_rot_sum)
        salmap_t = salmap_t.squeeze(0).squeeze(0)   # [H_out, W_out]

    # at this point we are out of no_grad, all big tensors are on CPU only

    # convert to numpy for further processing
    salmap = salmap_t.numpy()
    salmax_coords = np.unravel_index(np.argmax(salmap), salmap.shape)

    # normalize for visualization
    sal_min = salmap.min()
    sal_max = salmap.max()
    if sal_max > sal_min:
        salmap = (salmap - sal_min) / (sal_max - sal_min)
    else:
        salmap = np.zeros_like(salmap)
    salmap = (salmap * 255).astype(np.uint8)

    # explicitly delete large tensors and clear MPS cache
    del resized_frames, batch_frames, output_rot, output_rot_sum, salmap_t
    if device.type == "mps":
        torch.mps.empty_cache()

    return salmap, salmax_coords
