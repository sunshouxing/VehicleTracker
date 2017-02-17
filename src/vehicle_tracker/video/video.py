# -*- coding: utf-8 -*-

import abc
import cv2
import glob
import os

import util


@util.has_method('name', 'fps', 'read')
class AbstractVideo(object):
    """
    An abstract base class specifies the methods a subclass must have
    to be used as a video.
    """
    __metaclass__ = abc.ABCMeta


class Video(object):
    """
    Video is an abstraction of videos providing video's general info and
    a read method to get the next video frame.
    """

    def __init__(self, source):
        self.__video = cv2.VideoCapture(source)
        self.__name, _ = os.path.basename(source).rsplit('.')

        self.frame_num = 0

    def __del__(self):
        self.__video.release()

    def __str__(self):
        return ('Video general info:\n'
                '  o Name: {video.name};\n'
                '  o Fps: {video.fps};\n'
                '  o Frame width: {video.frame_width};\n'
                '  o Frame height: {video.frame_height};\n'
                '  o Frame count: {video.frame_count};\n'
                '  o Duration: {video.duration}.').format(video=self)

    def read(self):
        """
        Read a frame from this video, return the frame sequence number and
        the frame if success, return -1 and None if failed.
        """
        while self.__video.isOpened():
            status, frame = self.__video.read()
            if not status: break

            self.frame_num += 1
            yield self.frame_num, frame

    @property
    def name(self):
        return self.__name

    @property
    def fps(self):
        return self.__get(cv2.cv.CV_CAP_PROP_FPS)

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
    def duration(self):
        return self.frame_count / self.fps

    def __get(self, prop_id):
        return self.__video.get(prop_id)


class VideoCluster(object):
    """
    For unknown reason, some traffic surveillance video was split into several
    pieces, and the video pieces' names are like：
        20161114083000_20161114090000_P000.mp4
        20161114083000_20161114090000_P000(1).mp4
        20161114083000_20161114090000_P000(2).mp4
        ...
    The VideoCluster make it possible that using these video pieces just like
    a single video file.

    Demo:
        video = VideoCluster('20161114083000_20161114090000_P000.mp4')
        video_player = VideoPlayer()
        video_player.play(video)
    """

    def __init__(self, video):
        if not os.path.isfile(video):
            raise RuntimeError("the video file doesn't exist: {}".format(video))

        video_home = os.path.dirname(video)
        video_file = os.path.basename(video)

        self.start_time = util.timestamp(video_file[:12], '%Y%m%d%H%M')
        if self.start_time is None:
            raise RuntimeError('illegal video name: {}'.format(video_file))

        self.frame_num = 0

        # find all files like 'video_name*.mp4' in video home
        pattern = os.path.join(video_home, video_file.replace('.', '*.'))
        # sort the found files by the serial number
        sources = sorted(glob.glob(pattern),
                         key=lambda x: int(os.path.basename(x)[35:-5] or '0'))

        self.__videos = [Video(source) for source in sources]
        self.__name, _ = video_file.rsplit('.')

    def __del__(self):
        while self.__videos:
            video = self.__videos.pop()
            del video

    def read(self):
        for video in self.__videos:
            for _, frame in video.read():
                self.frame_num += 1
                yield self.frame_num, frame

    @property
    def fps(self):
        return self.__videos[0].fps

    @property
    def name(self):
        return self.__name

# EOF
