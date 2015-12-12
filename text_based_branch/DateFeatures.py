__author__ = 'Natawut'

import datetime

"""
This module contains functions to extract date features from reviews, specifically
whether or not the review was made on a weekday (Monday-Thursday).
"""


def parse_date(date):
    year, month, date = map(int, date.split('-'))
    return datetime.date(year, month, date)


def is_weekday(review):
    return parse_date(review.date).weekday() < 4