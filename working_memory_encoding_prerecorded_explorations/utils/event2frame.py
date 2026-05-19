import argparse
import os
import tqdm
import cv2
import numpy as np

from dat_files import load_dat_event


def convert_file(filename, output_dir, res, tw, timesurface=False, display=False):
    """
    Convert event file to frames
    :param filename: input event file
    :param res: resolution [width, height]
    :param tw: time window in microseconds
    :param timesurface: if True, show timesurface instead of decaying events
    :return:
    """
    ts, x, y, p = load_dat_event(filename)

    img         = np.zeros((res[1], res[0]), dtype=float)
    tsurface    = np.zeros((res[1], res[0]), dtype=np.int64)
    indsurface  = np.zeros((res[1], res[0]), dtype=np.int8)

    for i, t in enumerate(range(ts[0], ts[-1], tw)):
        # Get events in the current time window
        ind = np.where((ts > t) & (ts < t + tw))

        # Create a matrix holding the time stamps of the events
        tsurface[:, :] = 0
        tsurface[y[ind], x[ind]] = t + tw

        # And another holding their polarity (use -1 for OFF events)
        indsurface[y[ind], x[ind]] = 2.0 * p[ind] - 1

        # Find which pixels to process
        ind = np.where(tsurface > 0)

        # And update the image
        if timesurface:
            img[:, :] = 125
            img[ind]  = 125 + indsurface[ind] * np.exp(-(t + tw - tsurface[ind].astype(np.float32))/ (tw/30)) * 125
        else:
            img[:, :] = 0
            img[ind] = 255

        # save the image to the output_dir
        cv2.imwrite(os.path.join(output_dir, "frame_{:06d}.png".format(i)), img)

        # Convert to color and display
        if display:
            img_c = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            img_c = cv2.putText(img_c, '{} us'.format(t + tw), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255))
            img_c = cv2.applyColorMap(img_c, cv2.COLORMAP_VIRIDIS)
            cv2.imshow("debug", img_c)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break


def convert_dir(dir, output_dir, res, tw, timesurface=False, display=False):
    for filename in tqdm.tqdm(os.listdir(dir)):
        if filename.endswith(".dat"):
            # get the name of the object
            # from this format: event_android_100_10_100_40_0.4_0.01.dat
            object_name = filename.split("_")[1]
            sequence_number = filename.split("_")[2]
            # make folder for object in output_dir
            object_dir = os.path.join(output_dir, object_name, sequence_number)
            os.makedirs(object_dir, exist_ok=True)
            convert_file(os.path.join(dir, filename), object_dir, res, tw, timesurface, display)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_file", type=str)
    args.add_argument("--input_dir", type=str)
    args.add_argument("--output_dir", type=str)
    args.add_argument("--time_window", type=int, default=1000)
    args.add_argument("--resolution", type=int, nargs=2, default=(400,400))
    args.add_argument("--timesurface", action='store_true')
    args.add_argument("--display", action='store_true')
    args = args.parse_args()

    if args.input_file is not None:
        convert_file(args.input_file, args.resolution, args.time_window, args.timesurface, args.display)

    if args.input_dir is not None:
        convert_dir(args.input_dir, args.output_dir, args.resolution, args.time_window, args.timesurface, args.display)
