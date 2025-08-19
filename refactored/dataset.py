import os
import numpy as np
import cv2
import torch
import numpy as np
import struct
from datetime import datetime


# Paths
img_width, img_height = 400, 400
root = '/media/matt/bigdata/DATA/CRIB/train_data/'
memory_save_path = '/media/matt/bigdata/DATA/CRIB/workingmemory/'
# create path for the working memory
os.makedirs(memory_save_path, exist_ok=True)


# ======= FIND ALL OBJECTS =======
def discover_all_objects(root):
    """Discover all object directories in the root folder"""
    objects = []
    sequences = []
    if os.path.exists(root):
        for item in os.listdir(root):
            item_path = os.path.join(root, item)
            for sequence in os.listdir(item_path):
                sequence_path = os.path.join(item_path, sequence)
                if os.path.isdir(sequence_path) and not sequence.startswith('.'):
                    # Check if directory has any image files
                    image_files = [os.path.join(sequence_path, f) for f in os.listdir(sequence_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    sequences.append(image_files)
                    if image_files:
                        objects.append(item)
    return sorted(objects), sorted(sequences)



# from: https://github.com/neuromorphicsystems/IEBCS
# https://github.com/neuromorphicsystems/IEBCS/blob/main/src/dat_files.py
def load_dat_event(filename, start=0, stop=-1, display=False):
    """ Load .dat events from file.
        Args:
            filename: Path of the .dat file
            start: starting timestamp (us)
            stop: if different than -1, last timestamp
            display: display file info
        Returns:
             ts, x, y, pol numpy arrays of timestamps, positions, and polarities
     """
    f = open(filename, 'rb')
    if f == -1:
        print("The file does not exist")
        return
    else:
        if display: print("Load DAT Events: " + filename)
    l = f.readline()
    all_lines = l
    while l[0] == 37:
        p = f.tell()
        if display: print(l)
        l = f.readline()
        all_lines = all_lines + l
    # f.close()
    all_lines = str(all_lines)
    # f = open(filename, 'rb')
    f.seek(p, 0)
    evType = np.uint8(f.read(1)[0])
    evSize = np.uint8(f.read(1)[0])
    p = f.tell()
    l_last = f.tell()
    if start > 0:
        t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        while t < start:
            p = f.tell()
            t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
            dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])

    if stop > 0:
        t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        while t < stop:
            l_last = f.tell()
            t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
            dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
    else:
        l_last = f.seek(0, 2)

    num_b = ((l_last - p) // int(evSize)) * 2
    f.close()
    data = np.fromfile(filename, dtype=np.uint32, count=num_b, offset=p)
    ts = data[::2]
    v = 0
    ind = all_lines.find("Version")
    if ind > 0:
        v = int(all_lines[ind+8])
    if v >= 2:
        x_mask = np.uint32(0x00007FF)
        y_mask = np.uint32(0x0FFFC000)
        pol_mask = np.uint32(0x10000000)
        x_shift = 0
        y_shift = 14
        pol_shift = 28
    else:
        x_mask = np.uint32(0x00001FF)
        y_mask = np.uint32(0x0001FE00)
        pol_mask = np.uint32(0x00020000)
        x_shift = 0
        y_shift = 9
        pol_shift = 17
    x = data[1::2] & x_mask
    x = x >> x_shift
    y = data[1::2] & y_mask
    y = y >> y_shift
    pol = data[1::2] & pol_mask
    pol = pol >> pol_shift
    if len(ts) > 0:
        if display:
            print("First Event: ", ts[0], " us")
            print("Last Event: ", ts[-1], " us")
            print("Number of Events: ", ts.shape[0])
    return ts, x, y, pol


def load_events(object_name, data_folder):
    # given a path to a .dat file, load the events in the .dat file
    events = []
    dat_file = f"{data_folder}/event_{object_name}_100_10_100_40_0.4_0.01.dat"
    if os.path.exists(dat_file):
        ts, x, y, pol = load_dat_event(dat_file)
        events.append((ts, x, y, pol))
    else:
        print(f"File {dat_file} does not exist.")
    return events

