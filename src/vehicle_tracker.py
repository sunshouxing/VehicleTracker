# -*- coding: utf-8 -*-

import argparse
import glob
import numpy as np
from os.path import basename
from time import strptime, mktime

import cv2

# pre-defined colors
COLOR_WHITE = (0, 0, 0)
COLOR_BLACK = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 200, 0)
COLOR_YELLOW = (0, 255, 255)


# special pixel points of road monitoring area selected to do perspective transform
reference_points = np.array([
    [
        [[696, 0], [840, 8], [974, 16], [1100, 26]],
        [[692, 54], [884, 68], [1052, 80], [1208, 92]],
        [[690, 124], [928, 140], [1134, 154], [1322, 174]],
        [[688, 344], [1068, 360], [1372, 376], [1606, 388]],
        [[700, 756], [1286, 736], [1688, 722], [1920, 720]],
    ],
    [
        [[476, 189], [604, 184], [724, 179], [839, 176]],
        [[478, 230], [631, 222], [778, 219], [918, 214]],
        [[484, 307], [698, 299], [890, 293], [1061, 287]],
        [[496, 427], [777, 410], [1022, 393], [1232, 381]],
        [[569, 903], [1094, 798], [1454, 725], [1701, 676]],
    ],
], dtype=np.uint16)


target_shapes = [(70, 120), (70, 180)]


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

    def __init__(self):
        self.vehicles = []

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
                    self.vehicles.append(Vehicle(time, lane, contour_area, center))

        for vehicle in self.vehicles[:]:
            if time - vehicle.last_update > 0.5:
                self.vehicles.remove(vehicle)
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
        for vehicle in self.vehicles:
            if (x / 70 + 1 == vehicle.lane) and (0 <= y - vehicle.trails[-1][1] <= 150):
                return vehicle

        return None

    @staticmethod
    def is_new_vehicle(coordinate):
        return coordinate[1] <= 50


class VideoCapture(object):
    """
    this class is used to replay captured video, and process every two frame to
    detect and track vehicles.
    """

    def __init__(self, source, start_time, direction):
        self.capture = cv2.VideoCapture(source)
        self.start_time = start_time
        self.direction = direction

        # acquire video general info
        self.fps = self.capture.get(cv2.cv.CV_CAP_PROP_FPS)
        self.frame_count = int(self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

        # add frames which wanna to display in this map
        self.frames_to_display = {}

        # create objects for frame processing
        self.subtractor = cv2.BackgroundSubtractorMOG2(500, 300, False)
        self.kernel = np.ones((5, 5), np.uint8)
        self.trans_matrix = self.generate_trans_matrix()

        # init vehicle tracker
        self.vehicle_tracker = VehicleTracker()

    def __del__(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def replay(self):
        frame_num = 0

        while self.capture.isOpened():
            # read and process frames one by one
            result, frame = self.capture.read()

            if result:
                current_time = self.start_time + frame_num/25.0
                frame_num += 1

                # to improve performance, we only analysis every two frames instead of every frame
                if (frame_num % 2) == 0:
                    self.process(frame, current_time)
                    self.display()
            else:
                # failed to acquire the frame, then break the while loop
                break

            # video play control
            # key = cv2.waitKey(10) & 0xFF
            # if key == ord('q'):
            #     break
            # elif key == ord(' '):
            #     cv2.waitKey()
            # elif key == ord('s'):
            #     cv2.imwrite('background.jpg', frame)
            # else:
            #     pass

        return self.frame_count/self.fps

    def process(self, frame, time):
        # transform the frame to make analysis much easier
        transformed = self.prepare(frame)

        # background segmentation
        mask = self.bg_segment(transformed)

        contours, _ = cv2.findContours(mask, cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_NONE)
        contours = [cv2.convexHull(c) for c in contours]
        # cv2.drawContours(self.frames_to_display['gray'], contours, -1, COLOR_RED, 2)

        self.vehicle_tracker.detect(contours, time)

    def prepare(self, frame):
        # resize the frame
        # frame = cv2.resize(frame, (self.frame_width/2, self.frame_height/2))

        # convert frame to gray one
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # gaussian blurring to smooth our images
        # frame = cv2.GaussianBlur(frame, (11, 11), 0)

        # perspective transform
        frame = self.perspective(frame)
        # self.frames_to_display['gray'] = frame

        return frame

    def bg_segment(self, frame):
        mask = self.subtractor.apply(frame)

        # morphology operations
        morphology = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), dtype=np.uint8))

        return morphology

    def perspective(self, frame):
        """
        perspective transform to given frame
        """
        result = np.empty((600, 210), np.uint8)

        x, y = 0, 0
        for i in range(0, 12):
            shape = target_shapes[(i % 6 / 3 + self.direction) % 2]
            width, height = shape

            x = i % 3 * width
            target = result[y:(y + height), x:(x + width)]
            cv2.warpPerspective(frame, self.trans_matrix[i], shape, target)
            y += (i % 3 / 2) * height

        return result

    def generate_trans_matrix(self):
        matrix = np.empty((12, 3, 3), dtype=np.float64)

        for i in range(0, 12):
            x, y = (i / 3, i % 3)
            # select source reference points by traffic direction
            src_points = reference_points[self.direction][x:x + 2, y:y + 2].reshape((1, 4, 2))[0]
            # generate destination points according to traffic direction
            width, height = target_shapes[(i % 6 / 3 + self.direction) % 2]
            dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]])

            matrix[i] = cv2.getPerspectiveTransform(
                src_points.astype(np.float32), dst_points.astype(np.float32))

        return matrix

    def display(self):
        for window_name, frame in self.frames_to_display.items():
            cv2.imshow(window_name, frame)


def main():
    parser = argparse.ArgumentParser(
        description='This is a traffic monitoring system to detect and track vehicle by opencv',
    )
    parser.add_argument('-s', '--video-source', action='store', dest='video_source',
                        help='the traffic video source')
    parser.add_argument('-d', '--direction', action='store', dest='direction', type=int, default=0,
                        help='traffic direction: 0/1')
    arguments = parser.parse_args()

    video_source = arguments.video_source
    start_time = mktime(strptime(basename(video_source)[:12], '%Y%m%d%H%M'))

    video_sources = glob.glob(video_source.replace('.', '*.'))
    video_sources = video_sources[-1:] + video_sources[:-1]

    for video_source in video_sources:
        # initialize video capture and start it
        capture = VideoCapture(video_source, start_time, arguments.direction)
        time_cost = capture.replay()
        start_time += time_cost

if __name__ == '__main__':
    main()

# EOF
