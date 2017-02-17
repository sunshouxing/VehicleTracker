# -*- coding: utf-8 -*-

import cv2

try:
    from collections import ChainMap
except ImportError:
    from chainmap import ChainMap


def timestamp(string, format):
    """
    Parse a string to seconds since the Epoch according to a format specification.

    :param string: time string to parse
    :param format: time format specification
    :return: seconds since the Epoch if success, None if failed
    """
    from time import mktime, strptime

    try:
        ts = mktime(strptime(string, format))
    except ValueError:
        return None
    else:
        return ts


def has_method(*methods):
    """
    A decorator factory generates decorators used to decorate a base class, and
    the subclasses of the decorated base class must have all the given methods.
    """

    def decorator(base_class):
        def __subclasshook__(cls, subclass):
            if cls is base_class:
                attributes = ChainMap(*(
                    superclass.__dict__ for superclass in subclass.__mro__
                ))
                if all(method in attributes for method in methods):
                    return True
            return NotImplemented

        base_class.__subclasshook__ = classmethod(__subclasshook__)
        return base_class

    return decorator


def draw_line(image, start_point, end_point):
    """
    Draw a line with emphasized endpoint on given image.
    """

    cv2.circle(
        img=image,
        center=start_point,
        radius=5,
        color=(0, 0, 255),
        thickness=-1,
    )
    cv2.circle(
        img=image,
        center=end_point,
        radius=5,
        color=(0, 0, 255),
        thickness=-1,
    )
    cv2.line(
        img=image,
        pt1=start_point,
        pt2=end_point,
        color=(0, 0, 255),
        thickness=1,
    )

# EOF
