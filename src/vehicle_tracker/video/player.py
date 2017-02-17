# -*- coding: utf-8 -*-

import cv2

from video import AbstractVideo


class VideoPlayer(object):
    """
    A simple video player with keyboard control and optional self-define
    mouse callback.

    Keyboard control:
        * ESC:
            stop and quit video player
        * Space:
            pause video player and press any key to continue
        * U/u:
            speed up video player two times
        * D/d:
            slow down video player half
        * S/s:
            capture current frame

    Demo:
    def mouse_callback_func(event, x, y, flags, param):
        ...

    video = Video('20161114083000_20161114090000_P000.mp4', 1479083400)
    video_player = VideoPlayer(mouse_callback=_mouse_callback_func)
    video_player.play(video)
    """

    def __init__(self, mouse_callback=None):
        self.__mouse_callback = mouse_callback

        # variables for play control
        self.__fps = 25
        self.__frame_counter = 0
        self.__current_frame = None
        self.__exit = False

    def play(self, video):
        """
        Play given video.
        """
        # check the type of video parameter
        if not isinstance(video, AbstractVideo):
            raise TypeError("Expected object of type AbstractVideo, got {}".
                            format(type(video).__name__))

        # init video display window and set mouse callback if exist
        cv2.namedWindow(video.name, cv2.WINDOW_NORMAL)
        if self.__mouse_callback:
            cv2.setMouseCallback(video.name, self.__mouse_callback)

        self.__fps = int(video.fps)

        for frame_num, frame in video.read():
            # return directly when the exit flag set True
            if self.__exit: break

            self.__frame_counter += 1
            self.__current_frame = frame

            # display the video frame
            cv2.imshow(video.name, frame)
            # keyboard display control
            self.__play_control()

        cv2.destroyWindow(video.name)

    def __play_control(self):
        key = cv2.waitKey(1000 / self.__fps) & 0xFF
        # 超时
        if key == 255:
            return
        # Esc stop and quit
        elif key == 27:
            self.__exit = True
        # Space pause
        elif key == 32:
            print u'播放暂停，按任意键继续！'
            cv2.waitKey()
        # S/s capture current frame
        elif key in (83, 115):
            filename = 'CaptureFrame_{}.jpg'.format(self.__frame_counter)
            cv2.imwrite(filename, self.__current_frame)
            print u'画面已捕获并保存至{}'.format(filename)
        # U/u speed up
        elif key in (85, 117):
            self.__fps *= 2
            print u'播放帧率: {}帧/秒'.format(self.__fps)
        # D/d slow down
        elif key in (68, 100):
            self.__fps /= 2
            print u'播放帧率: {}帧/秒'.format(self.__fps)
        # keys not defined
        else:
            print (
                u"播放控制键列表:\n"
                u"  按Esc键退出播放;\n"
                u"  按空格键暂停播放，再按一次任意键恢复播放;\n"
                u"  按U/u按当前播放速度的2倍播放;\n"
                u"  按D/d按当前播放速度的1/2播放;\n"
                u"  按S/s键捕获当前画面，按当前时刻保存为jpg文件.")

# EOF
