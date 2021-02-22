from nltk.classify import ClassifierI
from statistics import mode

class ScoreVecClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    @staticmethod
    def main():
        print("ScoreVecClassifier Main Method called")

    def predict(self, features):
        votes = []
        for c in self._classifiers:
            v = c.predict(features)
            votes.append(v[0])
        return int(mode(votes))

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.predict(features)
            votes.append(v[0])
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return round(conf, 3)

    def score(self, features, accuracy_dict):
        score = 0
        for c in self._classifiers:
            v = c.predict(features)
            acc_score = accuracy_dict.get(type(c).__name__)
            if v[0] == 2:
                score = score + (acc_score * 100)
            else:
                score = score - (acc_score * 100)
        return round(score, 3)