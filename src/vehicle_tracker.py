# -*- coding: utf-8 -*-

import sys
import argparse
import logging
import glob
import cv2
from os.path import basename
from time import strptime, mktime

from video import Video, VideoProcessor

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('vehicle_tracker')




class Vehicle(object):
    MIN_AREA = 1000
    MAX_AREA = 100000

    def __init__(self, arrival_time, lane, area, coordinate):
        super(Vehicle, self).__init__()

        self.arrival_time = arrival_time
        self.exit_time = arrival_time

        self.lane = lane
        self.max_area = area  # we use max area to decide vehicle type

        self.trails = [coordinate]
        self.last_update = arrival_time

    def __str__(self):
        return "{} {} {} {}".format(
            self.lane, int(self.arrival_time), self.mean_speed, self.type)

    def trail(self, area, time, coordinate):
        self.max_area = max(self.max_area, area)
        self.trails.append(coordinate)
        self.last_update = time

    @property
    def mean_speed(self):
        # return round(30 / (self.exit_time - self.arrival_time), 2)
        displacement = (self.trails[-1][1] - self.trails[0][1]) / 20.0
        return round(displacement / (self.last_update - self.arrival_time), 2)

    @property
    def type(self):
        if self.max_area < 20000:
            return 'C'
        elif 20000 <= self.max_area <= 50000:
            return 'B'
        else:
            return 'A'


class VehicleTracker(object):
    """
    VehicleTracker detect vehicles from given contours, which are acquired
    by process video frames with opencv, and keep track of the vehicles.
    """

    def __init__(self, videos, start_time, direction, interval, debug):
        self.__videos = [Video(source) for source in videos]
        self.__start_time = start_time

        self.__video_processor = VideoProcessor(direction, interval, debug)
        self.__vehicles = []

    def run(self):
        start_time, time_offset = (self.__start_time, 0)
        for video in self.__videos:
            start_time += time_offset

            for frame_num, candidates in self.__video_processor.analysis(video):
                self.detect(candidates, start_time + frame_num/video.fps)

            time_offset = video.duration

    def detect(self, contours, time):
        for contour in contours:
            # skip this contour if it's not available
            contour_area = cv2.contourArea(contour)
            if not self.validate(contour_area):
                continue

            # calculate the center point of the contour
            center = self.center_point(contour)
            # error cause by bad frame
            if not center:
                break

            vehicle = self.is_existing_vehicle(center)
            if vehicle:
                # update existed vehicle's position
                vehicle.trail(contour_area, time, center)
            else:  # if no existed vehicle adjoin the contour
                # estimate further if a new vehicle coming
                if self.is_new_vehicle(center):
                    lane = center[0] / 70 + 1
                    self.__vehicles.append(Vehicle(time, lane, contour_area, center))

        for vehicle in self.__vehicles[:]:
            if time - vehicle.last_update > 0.5:
                self.__vehicles.remove(vehicle)
                if len(vehicle.trails) >= 8 and vehicle.trails[-1][1] >= 500:
                    print vehicle

    @staticmethod
    def validate(contour_area):
        return Vehicle.MIN_AREA < contour_area < Vehicle.MAX_AREA

    @staticmethod
    def center_point(contour):
        """
        calculate the center point of given contour.
        """
        m = cv2.moments(contour)
        if m['m00'] == 0:
            return None

        return int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])

    def is_existing_vehicle(self, coordinate):
        """
        estimate if a contour is an existing vehicle by its coordinate.
        """

        x, y = coordinate
        for vehicle in self.__vehicles:
            if (x / 70 + 1 == vehicle.lane) and (0 <= y - vehicle.trails[-1][1] <= 150):
                return vehicle

        return None

    @staticmethod
    def is_new_vehicle(coordinate):
        return coordinate[1] <= 50


def video_start_time(video_source):
    try:
        start_time = mktime(strptime(basename(video_source)[:12], '%Y%m%d%H%M'))
    except ValueError:
        return None
    else:
        return start_time


def setup_parser():
    parser = argparse.ArgumentParser(
        description=(
            'This is a traffic monitoring system to detect and track \
            vehicles by opencv.'),
        version='1.0')
    parser.add_argument(
        'video_source',
        action='store',
        help='the traffic video source, e.g. 20161115093000_20161115100000_P000.mp4')
    parser.add_argument(
        'direction',
        action='store',
        choices=('upward', 'downward'),
        metavar='direction',
        help='traffic direction: upward/downward')
    parser.add_argument(
        '-i', '--interval',
        action='store',
        type=int,
        default=1,
        metavar='n',
        help='process every n frames of video to speed up analysis')
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        dest='debug',
        default=False,
        help='show intermediate result of video processing if debug on',
    )

    return parser


def main():
    # setup argument parser then parse arguments
    parser = setup_parser()
    args = parser.parse_args()

    log.debug(args)

    # try to parse the video's start time from video file name,
    # and log the error message and exit if failed.
    start_time = video_start_time(args.video_source)
    if start_time is None:
        log.error('illegal video source file name: video source file\'s name '
                  'should be like 20161115093000_20161115100000_P000.mp4')
        sys.exit(-1)

    # find all videos of given video source
    video_sources = glob.glob(args.video_source.replace('.', '*.'))
    video_sources = video_sources[-1:] + video_sources[:-1]

    # init vehicle tracker and run it
    vehicle_tracker = VehicleTracker(
        video_sources, start_time, args.direction, args.interval, args.debug)
    vehicle_tracker.run()


if __name__ == '__main__':
    main()

# EOF
