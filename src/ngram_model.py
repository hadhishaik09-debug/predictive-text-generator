from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n):
        self.n = n
        # map: context_tuple -> Counter(next_word -> count)
        self.ngrams = defaultdict(Counter)
        # keep unigram counts for fallback
        self.unigrams = Counter()

    def train(self, tokens):
        # populate unigrams
        for t in tokens:
            self.unigrams[t] += 1

        # build n-gram context counts
        for i in range(len(tokens) - self.n):
            context = tuple(tokens[i : i + self.n])
            next_word = tokens[i + self.n]
            self.ngrams[context][next_word] += 1

    def predict_with_counts(self, context_words, top_k=3):
        # convert text string → list of words
        if isinstance(context_words, str):
            context_words = context_words.strip().split()

        # clean empty tokens
        context_words = [w for w in context_words if w]

        # Try full context first → then smaller context
        max_k = min(self.n, len(context_words))
        for k in range(max_k, 0, -1):
            ctx = tuple(context_words[-k:])
            if ctx in self.ngrams and len(self.ngrams[ctx]) > 0:
                return self.ngrams[ctx].most_common(top_k)

        # fallback: unigram
        return self.unigrams.most_common(top_k)

    def predict(self, context_words, top_k=3):
        # convert input text to word tokens
        if isinstance(context_words, str):
            context_words = context_words.strip().split()

        preds = self.predict_with_counts(context_words, top_k=top_k)
        return [w for w, c in preds]
