# -*- coding: utf-8 -*-

import argparse
import glob
import sys
import logging
from os.path import basename
from time import strptime, mktime

from vehicle_tracker import VehicleTracker


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('vehicle_tracker')


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

    # init vehicle tracker and run it
    vehicle_tracker = VehicleTracker(
        video_sources, start_time, args.direction, args.interval, args.debug)
    vehicle_tracker.run()


if __name__ == '__main__':
    main()

# EOF
