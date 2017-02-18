# -*- coding: utf-8 -*-

import argparse
import logging

import tracker
import video

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('vehicle_tracker')


def setup_parser():
    """
    Setup the application's argument parser.

    :return: argument parser
    """

    parser = argparse.ArgumentParser(
        description=(
            'This is a traffic monitoring system to detect and track '
            'vehicles by opencv.'),
        version='1.0')
    parser.add_argument(
        'video_source',
        action='store',
        help='the video source, e.g.20161115093000_20161115100000_P000.mp4')
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

    traffic_video = video.VideoCluster(args.video_source)

    # init vehicle tracker and run it
    vehicle_tracker = tracker.VehicleTracker(
        traffic_video, args.direction, args.interval, args.debug)
    vehicle_tracker.run()


if __name__ == '__main__':
    main()

# EOF
