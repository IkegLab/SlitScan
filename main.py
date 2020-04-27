#!/usr/bin/env python

import numpy as np
import cv2
import sys

class TimeShiftFrame(object):
    """docstring for TimeShiftFrame."""

    def __init__(self, width, height, channel=3, history_length=128):
        super(TimeShiftFrame, self).__init__()
        self.FRAME_HISTORY_LEN = history_length
        self.frameWidth = width
        self.frameHeight = height
        self.channel = channel
        self.frame_history = np.zeros((self.FRAME_HISTORY_LEN, self.frameHeight, self.frameWidth, self.channel), dtype=np.uint8)
        self.frame_history_pointer = -1
        self.timeShiftMatrix = None
        self.frame = np.zeros((self.frameHeight, self.frameWidth, self.channel), dtype=np.uint8)
        self.isUpdated = False

    def addNewFrame(self, frame):
        assert len(frame.shape) == 3
        assert self.channel == frame.shape[2]
        if frame.shape[1] != self.frameWidth or frame.shape[0] != self.frameHeight:
            frame = cv2.resize(frame , (self.frameWidth, self.frameHeight))
        self.frame_history_pointer = (self.frame_history_pointer + 1) % self.FRAME_HISTORY_LEN
        self.frame_history[self.frame_history_pointer] = frame
        self.isUpdated = True

    def getFrame(self):
        if not self.isUpdated:
            return self.frame
        for r in range(self.frameHeight):
            for c in range(self.frameWidth):
                t_shift = self.timeShiftMatrix[r,c]
                t = self.frame_history_pointer - t_shift
                self.frame[r, c] = self.frame_history[t][r, c]
        self.isUpdated = False
        return self.frame


import argparse

time_shift_matrix_generator = {
    'slitscan': lambda w, h, d: np.tile(np.linspace(0, d, h, endpoint=False, dtype=int).reshape(1, -1).T, (1, w)),
    'random': lambda w, h, d: np.random.randint(d, size=(h, w))
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('video_file', type=str, nargs='?',
                        help='input video file (webcam without specify this option.)')
    parser.add_argument('--video-device', type=int, default=0,
                        help='video device ID. (default: 0)')
    parser.add_argument('--timeshift-type', type=str,
                        default=list(time_shift_matrix_generator.keys())[0],
                        choices=time_shift_matrix_generator.keys(),
                        help='type of timeshift.')
    parser.add_argument('--max-time-delay', type=int, default=60,
                        help='max time delay.')
    parser.add_argument('--frame-size', type=str,
                        default='640x360',
                        help='frame size formated as "460x360"')


    args = parser.parse_args()

    if args.video_file is None:
        cap = cv2.VideoCapture(args.video_device)
    else:
        cap = cv2.VideoCapture(args.video_file)

    frameWidth = int(args.frame_size.split('x')[0])
    frameHeight = int(args.frame_size.split('x')[1])
    maxTimeDelay = args.max_time_delay
    timeShiftFrame = None

    while(cap.isOpened()):
        ret, frame = cap.read()

        if timeShiftFrame is None:
            timeShiftFrame = TimeShiftFrame(frameWidth, frameHeight, frame.shape[2])
            timeShiftFrame.timeShiftMatrix = time_shift_matrix_generator[args.timeshift_type](frameWidth, frameHeight, maxTimeDelay)

        timeShiftFrame.addNewFrame(frame)

        cv2.imshow('frame', timeShiftFrame.getFrame())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
