from collections import defaultdict, Counter
import random

class MarkovModel:
    def __init__(self):
        # map word -> Counter(next_word -> count)
        self.chain = defaultdict(Counter)

    def train(self, tokens):
        for i in range(len(tokens) - 1):
            w = tokens[i]
            nw = tokens[i + 1]
            self.chain[w][nw] += 1

    def predict(self, word, top_k=1):
        if word not in self.chain or len(self.chain[word]) == 0:
            return ["no prediction"]
        # return top_k by counts
        most = self.chain[word].most_common(top_k)
        return [w for w, c in most]

    def predict_weighted(self, word):
        """
        Weighted random selection based on observed counts.
        """
        if word not in self.chain or len(self.chain[word]) == 0:
            return "no prediction"
        items = list(self.chain[word].items())  # [(w,count),...]
        words, weights = zip(*items)
        total = sum(weights)
        # sample using weights
        probs = [w / total for w in weights]
        return random.choices(words, weights=probs, k=1)[0]
