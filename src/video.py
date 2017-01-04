# -*- coding: utf-8 -*-

import cv2
import numpy as np


# pre-defined colors
COLOR_WHITE = (0, 0, 0)
COLOR_BLACK = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 200, 0)
COLOR_YELLOW = (0, 255, 255)


class Video(object):

    def __init__(self, source):
        self.__video = cv2.VideoCapture(source)
        self.__frame_num = 0

    def __del__(self):
        self.__video.release()

    def read(self):
        """
        read a frame from this video, return the frame sequencer number and
        the frame if success, return -1 and None if failed.
        """
        if self.__video.isOpened():
            status, frame = self.__video.read()
            if status:
                self.__frame_num += 1
                return self.__frame_num, frame

        return -1, None

    @property
    def fps(self):
        return int(self.__get(cv2.cv.CV_CAP_PROP_FPS))

    @property
    def frame_width(self):
        return int(self.__get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self):
        return int(self.__get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self):
        return int(self.__get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    @property
    def frame_num(self):
        return self.__frame_num

    @property
    def duration(self):
        return self.frame_count / self.fps

    def __get(self, prop_id):
        return self.__video.get(prop_id)


class VideoProcessor(object):
    """
    this class is used to replay captured video, and process every two frame to
    detect and track vehicles.
    """

    # special pixel points of road monitoring area selected to do perspective transform
    reference_points = np.array([
        [  # downward reference points
            [[696, 0], [840, 8], [974, 16], [1100, 26]],
            [[692, 54], [884, 68], [1052, 80], [1208, 92]],
            [[690, 124], [928, 140], [1134, 154], [1322, 174]],
            [[688, 344], [1068, 360], [1372, 376], [1606, 388]],
            [[700, 756], [1286, 736], [1688, 722], [1920, 720]],
        ],
        [  # upward reference points
            [[476, 189], [604, 184], [724, 179], [839, 176]],
            [[478, 230], [631, 222], [778, 219], [918, 214]],
            [[484, 307], [698, 299], [890, 293], [1061, 287]],
            [[496, 427], [777, 410], [1022, 393], [1232, 381]],
            [[569, 903], [1094, 798], [1454, 725], [1701, 676]],
        ]
    ], dtype=np.uint16)

    # perspective transform's target rectangle shape in form of (width, height)
    target_shapes = [(70, 120), (70, 180)]

    def __init__(self, direction, interval, debug):
        self.__direction = direction == 'upward' and 1 or 0
        self.__interval = interval
        self.__debug = debug

        # add frames which wanna to display in this cache
        # self.__display_cache = weakref.WeakValueDictionary()
        self.__display_cache = {}

        # create objects for frame processing
        self.subtractor = cv2.BackgroundSubtractorMOG2(500, 300, False)
        self.kernel = np.ones((5, 5), np.uint8)
        self.trans_matrix = self.generate_trans_matrix()

    def __del__(self):
        cv2.destroyAllWindows()

    def analysis(self, video):
        while True:
            # read and process frames one by one
            frame_num, frame = video.read()

            if frame_num > 0:
                # process every n frames instead of every frame to speed up analysis
                if frame_num % self.__interval:
                    candidates = self.process(frame)
                    yield (frame_num, candidates)

                    if self.__debug:
                        self.display()
            else:
                # failed to acquire the frame, then break the while loop
                break

            # video play control
            if self.__debug:
                key = cv2.waitKey(10) & 0xFF
                # press key [q] to quit
                if key == 113:
                    break
                # press [space] to pause
                elif key == 32:
                    cv2.waitKey()
                # press key [c] to capture current frame
                elif key == 99:
                    cv2.imwrite('background.jpg', frame)
                else:
                    pass

    def process(self, frame):
        # transform the frame to make analysis much easier
        transformed = self.prepare(frame)

        # background segmentation
        mask = self.bg_segment(transformed)

        contours, _ = cv2.findContours(mask, cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_NONE)
        contours = [cv2.convexHull(c) for c in contours]
        cv2.drawContours(self.__display_cache['gray'], contours, -1, COLOR_RED, 2)

        return contours

    def prepare(self, frame):
        # resize the frame
        # frame = cv2.resize(frame, (self.frame_width/2, self.frame_height/2))

        # convert frame to gray one
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # gaussian blurring to smooth our images
        # frame = cv2.GaussianBlur(frame, (11, 11), 0)

        # perspective transform
        frame = self.perspective(frame)
        self.__display_cache['gray'] = frame

        return frame

    def bg_segment(self, frame):
        mask = self.subtractor.apply(frame)

        # morphology operations
        morphology = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), dtype=np.uint8))

        return morphology

    def perspective(self, frame):
        """
        perspective transform to given frame
        """
        result = np.empty((600, 210), np.uint8)

        x, y = 0, 0
        for i in range(0, 12):
            shape = VideoProcessor.target_shapes[(i % 6 / 3 + self.__direction) % 2]
            width, height = shape

            x = i % 3 * width
            target = result[y:(y + height), x:(x + width)]
            cv2.warpPerspective(frame, self.trans_matrix[i], shape, target)
            y += (i % 3 / 2) * height

        return result

    def generate_trans_matrix(self):
        matrix = np.empty((12, 3, 3), dtype=np.float64)

        # select source reference points of traffic direction
        reference_points = VideoProcessor.reference_points[self.__direction]

        for i in range(0, 12):
            # target rectangle shape
            width, height = VideoProcessor.target_shapes[(i % 6 / 3 + self.__direction) % 2]

            x, y = (i / 3, i % 3)
            # slice the source area to transform
            src_points = reference_points[x:x + 2, y:y + 2].reshape((1, 4, 2))[0]
            # generate destination points according to target rectangle shape
            dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]])

            matrix[i] = cv2.getPerspectiveTransform(
                src_points.astype(np.float32), dst_points.astype(np.float32))

        return matrix

    def display(self):
        for window_name, frame in self.__display_cache.items():
            cv2.imshow(window_name, frame)

# EOF
