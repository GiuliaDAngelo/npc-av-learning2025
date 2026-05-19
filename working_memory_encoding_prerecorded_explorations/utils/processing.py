import numpy as np


###############################
# ##### Spike to Tensor ##### #
###############################
def spikeTensor(timestep, timeWindow, pixels, recordingData):
    sizeRow, sizeCol = pixels
    times, index = recordingData
    times = times*timeWindow

    binsPixel = np.arange(0, sizeRow*sizeCol+1, 1)
    binsTimes = np.arange(0, timestep+timeWindow, timeWindow)

    video = np.histogram2d(times, index, (binsTimes, binsPixel))[0]
    video = np.reshape(video, (binsTimes.size-1, sizeRow, sizeCol))

    return video


#######################
# ##### Pooling ##### #
#######################
def coarseGraining(video, dimCrop, dimCell, dimWindow, mfr=False):
    video = video[:, :dimCrop, :dimCrop].reshape(video.shape[0], dimCell, dimCrop//dimCell, dimCell, dimCrop//dimCell)
    video = video.sum(axis=(2, 4))

    videoBins = np.zeros((video.shape[0]//dimWindow+1, video.shape[1], video.shape[2]))
    for w in range(video.shape[0]//dimWindow):
        if mfr is False:
            videoBins[w] = video[dimWindow*w:dimWindow*(w+1)].sum(axis=0)
        elif mfr is True:
            videoBins[w] = video[dimWindow*w:dimWindow*(w+1)].sum(axis=0)

    return videoBins


############################
# ##### Spike to Gif ##### #
############################
"""
def spikeToGif(name, binsTimes, pixels, events, framerate):
    sizeRow, sizeCol = pixels
    eventsTime, eventsIndex = events[:, 0], events[:, 1]

    binsPixel = np.arange(0, (sizeRow*sizeCol)+1, 1)
    video = np.histogram2d(eventsTime, eventsIndex, (binsTimes, binsPixel))[0]
    video = np.reshape(video, (-1, sizeRow, sizeCol))

    for f in range(video.shape[0]):
        plt.figure()
        plt.imshow(video[f], 'gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'.{f}.png')
        plt.close()

    os.system(f'ffmpeg -hide_banner -loglevel error -framerate {framerate} -y -i .%d.png ../../image/evimo/{name}.gif')
    os.system(f'rm .*.png')

    return 0
"""


"""
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'../../image/core50/original-s{s+1}o{o+1}.mp4', fourcc, 20, (sizeRow, sizeCol))
for frame in videoSource:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)
out.release()
"""