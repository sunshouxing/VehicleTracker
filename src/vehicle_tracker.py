# -*- coding: utf-8 -*-

import logging
import cv2
from video import Video, VideoProcessor

log = logging.getLogger('vehicle_tracker')


class Vehicle(object):
    MIN_AREA = 1000
    MAX_AREA = 500000

    def __init__(self, arrival_time, lane, area, coordinate):
        super(Vehicle, self).__init__()

        self.arrival_time = arrival_time
        self.last_update = arrival_time
        self.lane = lane
        self.max_area = area  # we use max area to decide vehicle type
        self.x, self.y = coordinate
        self.trails = [coordinate]

    def __str__(self):
        return "{} {} {} {}".format(
            self.lane, int(self.arrival_time), self.mean_speed, self.type)

    def trail(self, time, area, coordinate):
        self.last_update = time
        self.max_area = max(self.max_area, area)
        self.trails.append(coordinate)
        self.x, self.y = coordinate

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
        self.__debug = debug

        self.__video_processor = VideoProcessor(direction, interval, debug)
        self.__vehicles = []

    def run(self):
        start_time, time_offset = (self.__start_time, 0)

        self.__videos.sort()
        for video in self.__videos:
            start_time += time_offset

            analysisor = self.__video_processor.analysis(video)
            # for frame_num, candidates in self.__video_processor.analysis(video):
            for frame_num, candidates in analysisor:
                self.detect(candidates, start_time + frame_num/video.fps)

                # send detected vehicles to video analysisor to draw trails for debug
                if self.__debug:
                    analysisor.send(self.__vehicles)

            time_offset = video.duration

    def detect(self, contours, time):
        for contour in contours:
            # skip this contour if it's not available
            area = cv2.contourArea(contour)
            if not self.validate(area):
                continue

            # calculate the center point of the contour
            center = self.center_point(contour)
            # error caused by bad frame
            if not center:
                break

            vehicle = self.is_existing_vehicle(center)
            if vehicle:
                # update existed vehicle's position
                vehicle.trail(time, area, center)
            else:  # if no existed vehicle adjoin the contour
                # estimate further if a new vehicle coming
                if self.is_new_vehicle(center):
                    lane = center[0] / 70 + 1
                    self.__vehicles.append(Vehicle(time, lane, area, center))

        for vehicle in self.__vehicles[:]:
            if time - vehicle.last_update > 0.5:
                self.__vehicles.remove(vehicle)

                if self.__debug:
                    print vehicle.trails

                if len(vehicle.trails) >= 8 and vehicle.y >= 300:
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
            if abs(x - vehicle.x) <= 5 and (0 <= y - vehicle.y <= 140):
                return vehicle

        return None

    @staticmethod
    def is_new_vehicle(coordinate):
        return coordinate[1] <= 50

# EOF
