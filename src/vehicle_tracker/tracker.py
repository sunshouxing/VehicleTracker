# -*- coding: utf-8 -*-

import cv2
import json
import logging
import multiprocessing as mp
import os

import numpy as np

import video

LANE_WIDTH = 140


class ImageAnalyzer(mp.Process):
    """
    ImageAnalyzer is a process used to analyze capture image to
    acquire vehicle attributes.
    """

    def __init__(self, name, queues, repeats, fps, distance):
        super(ImageAnalyzer, self).__init__(name=name)

        self.queues = queues
        self.repeats = repeats
        self.distance = distance
        self.fps = fps

    def run(self):
        while True:
            jobs = [queue.get() for queue in self.queues]
            times = [job[0] for job in jobs]
            images = [job[1] for job in jobs]

            try:
                if len(set(times)) > 1:
                    logging.error('images with different timestamps, detail: {}'.format(times))
                    break
                start_time = times[0]

                rectangle_groups = [self._recognize(image) for image in images]
                for (x1, y1, width1, height1, lane), \
                    (x2, y2, width2, height2, lane) \
                        in self._conjugate(*rectangle_groups):
                    pixel_delta = y2 - y1

                    speed = int((3.6*self.distance*self.repeats*self.fps)/pixel_delta)
                    length = self.distance*height2/pixel_delta
                    time = int(start_time + (y2 / self.fps / self.repeats))

                    logging.info('{} {} {} {}'.format(lane, time, speed, length))
            finally:
                for queue in self.queues:
                    queue.task_done()

    def _recognize(self, image):
        """
        Recognize vehicles in capture image as rectangles.
        """
        # repeat each row 3 times to make vehicle recognition easier
        image = np.repeat(image, self.repeats, 0)
        # morphology transformation
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((4, 4), dtype=np.uint8))

        # vehicle recognition
        contours, _ = cv2.findContours(image, cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_NONE)
        rectangles = [cv2.boundingRect(c) for c in contours]

        return rectangles

    @staticmethod
    def _preprocess(rectangles):
        """
        1. filter the rectangles which are too small;
        2. add the lane info
        3. convert into a np.array
        """

        def lane_of(rectangle):
            x, y, width, height = rectangle
            if x < 0.8 * LANE_WIDTH:
                return 1
            elif (x > 1.8 * LANE_WIDTH) and (x + width > 2.8 * LANE_WIDTH):
                return 3
            else:
                return 2

        customized_type = np.dtype([
            ('x', np.uint16),
            ('y', np.uint16),
            ('width', np.uint16),
            ('height', np.uint16),
            ('lane', np.uint16),
        ])

        rectangles = filter(lambda a: a[2] > 60 and a[3] > 15, rectangles)
        return np.array(
            [(i[0], i[1], i[2], i[3], lane_of(i)) for i in rectangles],
            dtype=customized_type
        )

    def _conjugate(self, group1, group2):
        """
        Conjugate rectangles in group1(from line capture 1)
        and group2 (from line capture 2).
        """
        group1 = self._preprocess(group1)
        group2 = self._preprocess(group2)

        for item in group2:
            x, y, width, height, lane = item

            # find the rectangles having the same lane
            index = group1['lane'] == lane
            filtered = group1[index]
            # find the rectangles with larger y than the item's
            index = filtered['y'] < y
            filtered = filtered[index]

            # if there is no item matched
            if len(filtered) == 0: continue

            matched = filtered[0]

            # the speed in range (25, 150)
            # if 100 > item[1] - matched[1] > 10:
            #     yield matched, item
            yield matched, item


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

        # load overlays' configuration
        conf_file = os.path.join(os.path.dirname(__file__), 'overlays.json')
        with open(conf_file, 'r') as source:
            overlay_conf = json.load(source)

        # video processor => [job queues] => image analyzer
        overlays_num = len(overlay_conf['overlays'])
        self.job_queues = [mp.JoinableQueue() for _ in range(overlays_num)]

        self.video_processor = video.VideoProcessor(plugins=[
            video.plugins.OverlayCapturePlugin(direction, [
                video.plugins.OverlayCapture(queue, **conf)
                for queue, conf in zip(self.job_queues, overlay_conf['overlays'])
            ]),
        ])

        self.image_analyzer = ImageAnalyzer(
            'ImageAnalyzer', self.job_queues, 3, self.video.fps, overlay_conf['distance'])

    def run(self):
        self.image_analyzer.start()
        self.video_processor.process(self.video)

        self.image_analyzer.join()

# EOF
