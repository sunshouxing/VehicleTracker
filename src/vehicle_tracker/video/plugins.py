# -*- coding: utf-8 -*-

import abc
import functools
import cv2
import numpy as np


class VideoPlugin(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.active = True

    @abc.abstractmethod
    def process(self, frame, frame_num, timestamp):
        pass

    @abc.abstractmethod
    def finalize(self):
        pass


class DisplayPlugin(VideoPlugin):

    def __init__(self, window_name, fps):
        super(DisplayPlugin, self).__init__()

        self.window_name = window_name
        self.fps = fps
        self.current_frame = None
        self.frame_counter = 0

        self.CTRL_KEYS_PROMPT = (
            u"播放控制键列表:\n"
            u"  * 按Esc键退出播放;\n"
            u"  * 按空格键暂停播放，再按一次任意键恢复播放;\n"
            u"  * 按→按当前播放速度的2倍播放;\n"
            u"  * 按←按当前播放速度的1/2播放;\n"
            u"  * 按c键捕获当前画面，按当前时刻保存为jpg文件.")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def process(self, frame, frame_num, timestamp):
        if not self.active: return

        self.current_frame = frame
        self.frame_counter += 1

        cv2.imshow(self.window_name, frame)
        self.__play_control(timestamp)

    def finalize(self):
        cv2.destroyWindow(self.window_name)

    def __play_control(self, timestamp):
        key = cv2.waitKey(1000 / self.fps) & 0xFF
        # timeout
        if key == 255:
            pass
        # Esc
        elif key == 27:
            self.active = False
            self.finalize()
        # Space
        elif key == 32:
            print u'播放暂停，按任意键继续！'
            cv2.waitKey()
        # c
        elif key == 99:
            filename = 'CaptureFrame_{}.jpg'.format(int(timestamp))
            cv2.imwrite(filename, self.current_frame)
            print u'画面已捕获并保存至{}'.format(filename)
        # →
        elif key == 2555904:
            self.fps *= 2
            print u'播放帧率: {}帧/秒'.format(self.fps)
        # ←
        elif key == 2424832:
            self.fps /= 2
            print u'播放帧率: {}帧/秒'.format(self.fps)
        else:
            print self.CTRL_KEYS_PROMPT


class CaptureImage(object):

    def __init__(self, shape, image_queue):
        self.rows, self.columns = shape
        self.image_queue = image_queue

        self.image = np.empty(shape, dtype=np.uint8)
        self.index = 0
        self.timestamp = 0

    def concat(self, other, timestamp):
        if self.index == 0:
            self.timestamp = timestamp

        start, stop = self.index, self.index + other.shape[0]
        self.image[start:stop, :] = other

        # adjust the insert index value
        self.index += other.shape[0]
        # when the capture image is full
        # save it and then reset the insert index to 0
        if stop == self.rows:
            self.save()
            self.index = 0

    def save(self):
        if self.index == 0: return

        image = self.image[:self.index, :]
        self.image_queue.put((self.timestamp, image.copy()))


class OverlayCapture(object):

    def __init__(self, image_queue, name, start_point, end_point, size):
        self.name = name

        # init slicer
        x1, y1 = start_point
        x2, y2 = end_point
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        self.slicer = np.s_[y1:y2 + 1, x1:x2 + 1]

        # init capture image
        image_shape = (y2-y1+1)*size, (x2-x1+1)
        self.capture_image = CaptureImage(image_shape, image_queue)

        # init subtractor
        self.subtractor = cv2.BackgroundSubtractorMOG()

    def capture(self, frame, timestamp):
        frame_slice = frame[self.slicer]
        frame_mask = self.subtractor.apply(frame_slice, learningRate=1.0/100)
        self.capture_image.concat(frame_mask, timestamp)


class Transformer(object):
    """
    This class is used to perspective transform video frame.
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
    target_shapes = [(140, 240), (140, 360)]

    def __init__(self, direction, debug):
        self.__direction = direction == 'upward' and 1 or 0
        self.__debug = debug

        self.__trans_matrix = self.__generate_trans_matrix()

        # create objects for frame processing
        # self.subtractor = cv2.BackgroundSubtractorMOG2()
        self.subtractor = cv2.BackgroundSubtractorMOG()
        self.kernel = np.ones((5, 5), np.uint8)

    def perspective(self, frame):
        """
        perspective transform to given frame
        """
        shapes = Transformer.target_shapes
        height = (shapes[0][1] + shapes[1][1]) * 2
        width = shapes[0][0] * 3
        result = np.empty((height, width), np.uint8)

        x, y = 0, 0
        for i in range(0, 12):
            shape = Transformer.target_shapes[(i % 6 / 3 + self.__direction) % 2]
            width, height = shape

            x = i % 3 * width
            target = result[y:(y + height), x:(x + width)]
            cv2.warpPerspective(frame, self.__trans_matrix[i], shape, target)
            y += (i % 3 / 2) * height

        return result

    def __generate_trans_matrix(self):
        matrix = np.empty((12, 3, 3), dtype=np.float64)

        # select source reference points of traffic direction
        reference_points = Transformer.reference_points[self.__direction]

        for i in range(0, 12):
            # target rectangle shape
            width, height = Transformer.target_shapes[(i % 6 / 3 + self.__direction) % 2]

            x, y = (i / 3, i % 3)
            # slice the source area to transform
            src_points = reference_points[x:x + 2, y:y + 2].reshape((1, 4, 2))[0]
            # generate destination points according to target rectangle shape
            dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]])

            matrix[i] = cv2.getPerspectiveTransform(
                src_points.astype(np.float32), dst_points.astype(np.float32))

        return matrix


class OverlayCapturePlugin(VideoPlugin):

    def __init__(self, direction, overlays):
        super(OverlayCapturePlugin, self).__init__()

        self.overlays = overlays

        self.preprocess_routines = [
            # convert frame to gray one
            functools.partial(cv2.cvtColor, code=cv2.COLOR_BGR2GRAY),
            # perspective transform
            Transformer(direction, False).perspective,
            # gaussian blur
            functools.partial(cv2.GaussianBlur, ksize=(21, 21), sigmaX=0),
        ]

    def process(self, frame, frame_num, timestamp):
        if not self.active: return

        if self.overlays:
            for routine in self.preprocess_routines:
                frame = routine(frame)

            for overlay in self.overlays:
                overlay.capture(frame, timestamp)

    def finalize(self):
        for overlay in self.overlays:
            overlay.capture_image.save()

# EOF
