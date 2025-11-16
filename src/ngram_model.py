from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n):
        self.n = n
        # map: context_tuple -> Counter(next_word -> count)
        self.ngrams = defaultdict(Counter)
        # also keep unigram counts for fallback
        self.unigrams = Counter()

    def train(self, tokens):
        # populate unigrams
        for t in tokens:
            self.unigrams[t] += 1

        # create contexts of length self.n and track next word counts
        for i in range(len(tokens) - self.n):
            context = tuple(tokens[i : i + self.n])
            next_word = tokens[i + self.n]
            self.ngrams[context][next_word] += 1

    def predict_with_counts(self, context_words, top_k=3):
        """
        Try full context (length self.n), fall back to smaller contexts,
        then fall back to unigram.
        Returns list of (word, count) tuples sorted by count desc.
        """
        # normalize context input
        context_words = [w for w in context_words if w]
        # try contexts of decreasing size
        for k in range(self.n, 0, -1):
            if len(context_words) >= k:
                ctx = tuple(context_words[-k:])
                if ctx in self.ngrams and len(self.ngrams[ctx]) > 0:
                    most = self.ngrams[ctx].most_common(top_k)
                    return most

        # fallback: return top unigrams
        return self.unigrams.most_common(top_k)

    def predict(self, context_words, top_k=3):
        preds = self.predict_with_counts(context_words, top_k=top_k)
        return [w for w, c in preds]
