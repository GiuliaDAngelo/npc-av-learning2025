import numpy as np 
import cv2

class EventFrameRendererWindow:
    # rendering just the latest timewindow of events
    def __init__(self, width, height, tau):
        self.width, self.height = width, height
        self.window_us = tau
        self.now = 0

    def update(self, events, dt_us):
        self.now += dt_us

        # pozadí = 128 (žádná událost)
        img = np.full((self.height, self.width), 0, dtype=np.uint8)

        if events.i > 0:
            mask = events.ts[:events.i] >= (self.now - self.window_us)
            x = events.x[:events.i][mask]
            y = events.y[:events.i][mask]
            p = events.p[:events.i][mask]

            # do not distinguish polarity of events
            img[y[p == 1], x[p == 1]] = 255   # ON
            img[y[p == 0], x[p == 0]] = 255    

            # only positive (ON) events
            #on = (p == 1)
            #img[y[on], x[on]] = 255


        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)