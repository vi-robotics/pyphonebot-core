#!/usr/bin/env python3

import numpy as np
from threading import Thread
from multiprocessing import Process, Queue


def spawn_opencv_key_window(queue: Queue):
    """
    Spawn an opencv window just to receive keyboard events.
    Internally imports cv2 to prevent enforcing unnecessary dependency.
    """
    def spin_opencv_window(queue: Queue):
        import cv2
        win = cv2.namedWindow('key_win')
        dummy = np.zeros((512, 512), dtype=np.uint8)
        cv2.imshow('key_win', dummy)

        while True:
            k = cv2.waitKey(0)
            if k in [ord('q'), 27]:
                break
            if not queue.full():
                queue.put_nowait(chr(k))

    p = Process(target=spin_opencv_window, args=[queue])
    p.daemon = True
    p.start()
