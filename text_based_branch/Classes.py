__author__ = 'Natawut'


class Review(object):
    def __init__(self, dict):
        self.votes = dict["votes"]
        self.user_id = dict["user_id"]
        self.review_id = dict["review_id"]
        self.stars = dict["stars"]
        self.date = dict["date"]
        self.text = dict["text"]
        self.business_id = dict["business_id"]
        self.features = []