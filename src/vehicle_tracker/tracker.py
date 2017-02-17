# -*- coding: utf-8 -*-

import video


class Vehicle(object):
    """

    """
    pass


class VehicleTracker(object):
    """
    VehicleTracker is designed to detect and track the vehicles by analyzing the
    traffic surveillance video.
    """

    def __init__(self, traffic_video, direction, interval, debug):
        self.video = traffic_video

        self.video_processor = video.VideoProcessor(plugins=[
            video.plugins.OverlayCapturePlugin()
        ])

    def run(self):
        self.video_processor.process(self.video)


# EOF
