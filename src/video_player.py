# -*- coding: utf-8 -*-
__author__ = 'SUN Shouwang'

import cv2
import time
import numpy as np


class Video(object):

    def __init__(self, src):
        self.video_capture = cv2.VideoCapture(src)
        self._fps_ = self.get(cv2.cv.CV_CAP_PROP_FPS).__int__()
        self.status = True
        self.frame = np.zeros((self.frame_height, self.frame_width, 1), dtype=np.uint8)
        self.frame_num = -1
        self.read()
        print \
            u'画面宽度: {self.frame_width}像素\n' \
            u'画面高度: {self.frame_height}像素\n' \
            u'帧数: {self.frame_count}\n' \
            u'播放帧率: {self.fps}帧/秒\n'.format(self=self)

    def get(self, prop_id):
        return self.video_capture.get(prop_id)

    def read(self):
        self.status, self.frame = self.video_capture.read()
        self.frame_num += 1

    @property
    def frame_width(self):
        return self.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH).__int__()

    @property
    def frame_height(self):
        return self.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT).__int__()

    @property
    def frame_count(self):
        return self.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT).__int__()

    @property
    def fps(self):
        return self._fps_

    @fps.setter
    def fps(self, value):
        if value > 0 and isinstance(value, int):
            self._fps_ = value
        else:
            raise TypeError(u'视频播放帧率只能为正整数')


class VideoPlayer(object):

    def __init__(self, video):

        self.video = video

        self.ctrl_keys = \
            u'播放控制键列表:\n' \
            u'按Esc键退出播放;\n' \
            u'按空格键暂停播放，再按一次任意键恢复播放;\n' \
            u'按→按当前播放速度的2倍播放;\n' \
            u'按←按当前播放速度的1/2播放;\n' \
            u'按L键开启鼠标拖拽添加辅助线;\n'\
            u'按S键捕获当前画面，按当前时刻保存为jpg文件.\n'
        print self.ctrl_keys

        self.show_overlays = True
        self.start_callback = False

        cv2.namedWindow('PlayWindow', cv2.WINDOW_NORMAL)

    def play_control(self):
        key = cv2.waitKey(1000/self.video.fps)
        # 超时
        if key is -1:
            pass
        # Esc键
        elif key is 27:
            self.video.status = False
        # 空格键
        elif key is ord(' '):
            print u'播放暂停，按任意键继续！'
            cv2.waitKey()
        # S或s键
        elif key in (83, 115):
            filename = 'CaptureImage_{}.jpg'.format(time.strftime('%Y%m%d%H%M%S'))
            cv2.imwrite(filename, self.video.frame)
            print u'画面已捕获并保存至%s' % filename
        # O或o键
        elif key in (79, 111):
            self.show_overlays = not self.show_overlays
            if self.show_overlays:
                print u'显示辅助图形'
            else:
                print u'隐藏辅助图形'
        # L或l键
        elif key in (76, 108):
            cv2.setMouseCallback('PlayWindow', OverlayLine.mouse_callback)
            print u'支持鼠标拖拽添加辅助线'
        # C或c键
        elif key in(67, 99):
            self.start_callback = not self.start_callback
            if self.start_callback:
                print u'开始捕获图像'
            else:
                print u'停止捕获图像'
        # →键
        elif key == 2555904:
            self.video.fps *= 2
            print u'当前帧率: %s帧/秒' % self.video.fps
        # ←键
        elif key == 2424832:
            self.video.fps /= 2
            print u'当前帧率: %s帧/秒' % self.video.fps
        # 无定义按键
        else:
            print self.ctrl_keys

    def play(self):
        capture_images = list()
        while self.video.status:
            if self.show_overlays:
                for instance in Overlay.get_instances():
                    instance.show(self.video.frame)
            cv2.imshow('PlayWindow', self.video.frame)
            self.video.read()
            self.play_control()
            if self.start_callback is True:
                play_callback(self.video, Overlay.get_instances(), capture_images)


class Overlay(object):

    _instances_ = list()

    def __new__(cls, *args, **kwargs):
        instance = super(Overlay, cls).__new__(cls, *args, **kwargs)
        cls._instances_.append(instance)
        return instance

    @classmethod
    def get_instances(cls):
        return cls._instances_

    def show(self):
        pass

    @classmethod
    def mouse_callback(cls):
        pass


class OverlayLine(Overlay):

    def __init__(self, start_point=None, end_point=None):
        self.start_point = start_point
        self.end_point = end_point
        self.confirmed = False

    def show(self, img):
        cv2.circle(
            img=img,
            center=self.start_point,
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
            lineType=cv2.LINE_AA)
        cv2.circle(
            img=img,
            center=self.end_point,
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
            lineType=cv2.LINE_AA)
        cv2.line(
            img=img,
            pt1=self.start_point,
            pt2=self.end_point,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA)

    @classmethod
    def mouse_callback(cls, event, x, y, flags, param):

        if not cls.get_instances():
            instance = OverlayLine()
        else:
            instance = cls.get_instances()[-1]

        # 按下左键
        if event is cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                instance = OverlayLine()
            instance.start_point = x, y
            instance.end_point = x, y

        # 按住左键并移动
        if event is cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    x0, y0 = instance.start_point
                    if abs(x-x0) > abs(y-y0):
                        instance.end_point = x, y0
                    else:
                        instance.end_point = x0, y
                else:
                    instance.end_point = x, y

        # 松开左键
        if event is cv2.EVENT_LBUTTONUP:
            instance.confirmed = True
            print '虚拟线{}: 起点{vl.start_point}，终点{vl.end_point}'.format(
                cls.get_instances().__len__(), vl=instance)

    def capture(self, img):

        x1, y1 = self.start_point
        x2, y2 = self.end_point

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        index = np.s_[x1:x2+1, y1:y2+1]
        # 使用切片[n, :]与[n:n+1, :]获得的内容虽然相同，
        # 但是[n:n+1, :]保留n:n+1对应的维度（尽管该维度的大小为1）

        return img[index]


def play_callback(video, overlay_lines, capture_images):

    if not capture_images:
        for i in range(overlay_lines.__len__()):
            capture = overlay_lines[i].capture(video.frame)
            image_shape = (1500, capture.size/3, 3)
            capture_images.append(np.zeros(image_shape, dtype=np.uint8))

    j = video.frame_num % 1500
    if j == 0:
        for i in range(capture_images.__len__()):
            cv2.imwrite('capture_image_{}_{}.jpg'.format(i, video.frame_num), capture_images[i])

    for i in range(capture_images.__len__()):
        capture_images[i][j:j+1] = overlay_lines[i].capture(video.frame)

    cv2.imshow('capture1', capture_images[0])


if __name__ == '__main__':
    video = Video('video/20161112080000_20161112083000_P000(2).mp4')
    video_player = VideoPlayer(video)
    video_player.play()
