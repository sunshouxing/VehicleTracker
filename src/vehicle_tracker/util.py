# -*- coding: utf-8 -*-

import cv2
import json

import numpy as np

from vehicle_tracker import colors
import video.util


def get_app_version():
    from version import __VERSION__
    return __VERSION__


def _draw_stat(image_source, stats):
    _colors = [colors.GREEN, colors.RED, colors.YELLOW]

    image = cv2.imread(image_source)
    image = np.repeat(image, 3, 0)

    with open(stats, 'r') as inputs:
        stats_data = json.load(inputs)

        for i, (x, y, width, height, lane, speed) in enumerate(stats_data):
            cv2.rectangle(image, (x, y), (x+width, y+height), _colors[lane-1], 2)
            cv2.putText(image, '{} | {}'.format(i, speed), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors.RED, 2)

    return image


def debug(time):
    image1 = _draw_stat('debug1-{}.jpg'.format(time), 'debug1-{}.json'.format(time))
    image2 = _draw_stat('debug2-{}.jpg'.format(time), 'debug2-{}.json'.format(time))

    image = np.column_stack((image1, image2))

    for i in range(1, 6):
        video.util.draw_line(image, (140 * i, 0), (140 * i, 4500), colors.YELLOW)

    cv2.imwrite('debug.jpg', image)

# EOF
