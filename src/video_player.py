# -*- coding: utf-8 -*-

import cv2
import numpy as np

__all__ = ['Video', 'VideoPlayer']


class Video(object):
    def __init__(self, source):
        self.video_capture = cv2.VideoCapture(source)
        self.frame_num = 0

        print self

    def __del__(self):
        self.video_capture.release()

    def __str__(self):
        return (
            '画面宽度: {self.frame_width}像素\n'
            '画面高度: {self.frame_height}像素\n'
            '帧数: {self.frame_count}\n'
            '播放帧率: {self.fps}帧/秒\n').format(self=self)

    @property
    def fps(self):
        return int(self.__get(cv2.cv.CV_CAP_PROP_FPS))

    @property
    def frame_count(self):
        return int(self.__get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    @property
    def frame_width(self):
        return int(self.__get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self):
        return int(self.__get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    @property
    def is_opened(self):
        return self.video_capture.isOpened()

    def read(self):
        """
        read a frame from this video, return the frame sequence number and
        the frame if success, return -1 and None if failed.
        """
        result, frame = self.video_capture.read()
        if result:
            self.frame_num += 1
            return self.frame_num, frame

        return -1, None

    def __get(self, prop_id):
        return self.video_capture.get(prop_id)


class VideoPlayer(object):
    def __init__(self, window=None, callback_fun=None, **callback_params):

        self.window = window

        self.callback_func = callback_fun
        self.callback_params = callback_params
        # 如果参数window与play_callback都未指定，创建一个默认窗口用于播放视频
        if self.window is None and self.callback_func is None:
            self.window = 'default_window'

        self.fps = 1
        self.current_frame = None
        self.acc_frame_num = 0

        self.exit = False

        self.CTRL_KEYS_PROMPT = u'播放控制键列表:\n' \
                                u'按Esc键退出播放;\n' \
                                u'按空格键暂停播放，再按一次任意键恢复播放;\n' \
                                u'按→按当前播放速度的2倍播放;\n' \
                                u'按←按当前播放速度的1/2播放;\n' \
                                u'按S键捕获当前画面，按当前时刻保存为jpg文件.\n'
        print self.CTRL_KEYS_PROMPT

    def play_control(self):
        key = cv2.waitKey(1000 / self.fps) & 0xFF
        # 超时
        if key == 255:
            pass
        # Esc键
        elif key == 27:
            self.exit = True
        # 空格键
        elif key == 32:
            print u'播放暂停，按任意键继续！'
            cv2.waitKey()
        # S或s键
        elif key in (83, 115):
            filename = 'CaptureFrame_{}.jpg'.format(self.acc_frame_num)
            cv2.imwrite(filename, self.current_frame)
            print u'画面已捕获并保存至{}'.format(filename)
        # →键
        elif key == 2555904:
            self.fps *= 2
            print u'播放帧率: {}帧/秒'.format(self.fps)
        # ←键
        elif key == 2424832:
            self.fps /= 2
            print u'播放帧率: {}帧/秒'.format(self.fps)
        # 无定义按键
        else:
            print self.CTRL_KEYS_PROMPT

    def play(self, video):
        self.fps = video.fps

        while video.is_opened:
            # return directly when the exit flag set True
            if self.exit: return

            frame_num, frame = video.read()
            if frame_num != -1:
                self.current_frame = frame
                self.acc_frame_num += 1

                if self.callback_func is not None:
                    self.callback_func(self.current_frame, self.acc_frame_num, **self.callback_params)

                if self.window is not None:
                    cv2.imshow(self.window, self.current_frame)
                    self.play_control()


class OverlayLine(object):
    __instances = []

    def __new__(cls, *args, **kwargs):
        instance = super(OverlayLine, cls).__new__(cls, *args, **kwargs)
        cls.__instances.append(instance)
        return instance

    @classmethod
    def get_instances(cls):
        return cls.__instances

    def __init__(self, start_point, end_point):
        self.start_point, self.end_point = start_point, end_point

        # calculate capture slice according to start_point and end_point
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        self.slice = np.s_[y1:y2 + 1, x1:x2 + 1]

        self.capture_image = None
        self.history = 1500

    def capture(self, frame, frame_num):
        frame_slice = frame[self.slice]

        # bg_subtractor = cv.createBackgroundSubtractorKNN(history=500, dist2Threshold=500)
        # capture = bg_subtractor.apply(img[index])

        # 第一种方法
        rows, cols, pages = frame_slice.shape
        if self.capture_image is None:
            if rows < cols:
                shape = (rows * self.history, cols, pages)
            else:
                shape = (rows, cols * self.history, pages)
            self.capture_image = np.zeros(shape, dtype=np.uint8)

        if rows < cols:
            self.capture_image[rows:, :, :] = self.capture_image[:-rows, :, :]
            self.capture_image[:rows, :, :] = frame_slice
        else:
            self.capture_image[:, cols:, :] = self.capture_image[:, :-cols, :]
            self.capture_image[:, :cols, :] = frame_slice

        # # 第二种方法
        # shape = capture.shape
        # if shape[0] < shape[1]:
        #     axis = 0
        # else:
        #     axis = 1
        # if self.capture_image is None:
        #     self.capture_image = np.repeat(np.zeros_like(capture), self.history, axis=axis)
        # self.capture_image[-shape[0]:, -shape[1]:, -shape[2]:] = capture
        # self.capture_image = np.roll(self.capture_image, shape[axis], axis=axis)

        if frame_num % 1500 == 0:
            # TODO Arthur format overlay line index
            file_name = 'CaptureImage_OverlayLine{}_FrameNum{}.jpeg'.format(1, frame_num)
            cv2.imwrite(file_name, self.capture_image)


# def mouse_callback(event, x, y, flags, param):
#     # 按下左键
#     if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
#         mouse_callback.start_point = x, y
#
#     # 按住左键并移动
#     if event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
#             if flags & cv2.EVENT_FLAG_SHIFTKEY:
#                 x0, y0 = mouse_callback.start_point
#                 if abs(x - x0) > abs(y - y0):
#                     y = y0
#                 else:
#                     x = x0
#
#             mouse_callback.end_point = x, y
#
#     # 松开左键
#     if event == cv2.EVENT_LBUTTONUP:
#         overlay = OverlayLine(mouse_callback.start_point, mouse_callback.end_point)
#         print '虚拟线: 起点{vl.start_point}，终点{vl.end_point}'.format(vl=overlay)
#
# mouse_callback.start_point = None
# mouse_callback.end_point = None


def play_callback(frame, frame_num, **callback_params):
    overlay_lines = OverlayLine.get_instances()

    for overlay_line in overlay_lines:
        # FIXME draw overlay line for debug
        draw_line(frame, overlay_line.start_point, overlay_line.end_point)
        overlay_line.capture(frame, frame_num)


def draw_line(image, start_point, end_point):
    cv2.circle(
        img=image,
        center=start_point,
        radius=5,
        color=(0, 0, 255),
        thickness=-1,
    )
    cv2.circle(
        img=image,
        center=end_point,
        radius=5,
        color=(0, 0, 255),
        thickness=-1,
    )
    cv2.line(
        img=image,
        pt1=start_point,
        pt2=end_point,
        color=(0, 0, 255),
        thickness=1,
    )


def main():
    cv2.namedWindow('PlayWindow', cv2.WINDOW_NORMAL)
    # cv2.setMouseCallback('PlayWindow', mouse_callback)

    cv2.namedWindow('CaptureImage', cv2.WINDOW_NORMAL)
    OverlayLine((657, 660), (1858, 660))
    video_player = VideoPlayer(
        window='PlayWindow',
        callback_fun=play_callback,
        capture_window='CaptureImage'
    )

    video = Video('/home/arthur/Workspace/VehicleTracker/app/video/20161114083000_20161114090000_P000.mp4')
    video_player.play(video)


if __name__ == '__main__':
    main()

# EOF
