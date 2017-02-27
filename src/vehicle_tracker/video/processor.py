# -*- coding: utf-8 -*-

import cv2
import logging


class VideoProcessor(object):
    """
    VideoProcessor implements a framework to read and process every frame of a
    video. The process method only provides a general routine of reading frames
    from video and sending the frames to video plugins for further processing,
    so implement a video plugin for your specific purpose, then the video
    processor will take care of other things.
    """

    def __init__(self, plugins):
        self.plugins = plugins

    def __del__(self):
        cv2.destroyAllWindows()

    def process(self, video):
        logging.info('Processing video {},\n{}'.format(video.name, video))

        for frame_num, frame in video.read():
            timestamp = video.start_time + frame_num/video.fps

            for plugin in self.plugins:
                if plugin.active: plugin.process(frame, frame_num, timestamp)

        for plugin in self.plugins:
            plugin.finalize()

        logging.info('Meets the end of video {}'.format(video.name))

# EOF
