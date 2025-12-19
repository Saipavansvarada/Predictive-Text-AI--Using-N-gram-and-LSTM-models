# models/ngram_predictor.py
from collections import defaultdict, Counter
import re
from typing import List, Tuple

def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"\b\w+\b", text)

class NGramPredictor:
    """
    Interpolated n-gram predictor (unigram + bigram + trigram by default)
    with prefix filtering for completion-style suggestions.
    """
    def __init__(self, n=3, alpha_unigram=0.1, alpha_bigram=0.3, alpha_trigram=0.6):
        assert n >= 1 and n <= 3
        self.n = n
        # counts: context (tuple) -> Counter(next_word)
        self.counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.total_unigrams = 0

        # interpolation weights (should sum to 1)
        # default favors trigram more when available
        self.alpha_unigram = alpha_unigram
        self.alpha_bigram = alpha_bigram
        self.alpha_trigram = alpha_trigram

    def train(self, texts: List[str]):
        for line in texts:
            tokens = tokenize(line)
            if not tokens:
                continue
            # update unigrams
            for t in tokens:
                self.unigram_counts[t] += 1
                self.total_unigrams += 1

            # add sentence boundary tokens for context
            toks = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            L = len(toks)
            for i in range(L - 0):
                # build contexts up to n-1
                for ctx_len in range(0, self.n):
                    if i + ctx_len + 1 > L:
                        break
                    context = tuple(toks[i:i+ctx_len])  # length ctx_len (0..n-1)
                    next_idx = i + ctx_len
                    if next_idx < L:
                        nxt = toks[next_idx]
                        self.counts[(ctx_len, context)][nxt] += 1

    def _prob_unigram(self, word: str) -> float:
        # simple MLE + add-epsilon smoothing to avoid zero probs
        eps = 1e-9
        return (self.unigram_counts.get(word, 0) + eps) / (self.total_unigrams + eps * (len(self.unigram_counts) + 1))

    def _prob_bigram(self, context_word: Tuple[str], word: str) -> float:
        # context_word is a tuple length 1
        ctx_key = (1, context_word)
        cnt = self.counts.get(ctx_key, None)
        if not cnt:
            return 0.0
        total = sum(cnt.values())
        return cnt.get(word, 0) / total

    def _prob_trigram(self, context_words: Tuple[str], word: str) -> float:
        # context_words length 2
        ctx_key = (2, context_words)
        cnt = self.counts.get(ctx_key, None)
        if not cnt:
            return 0.0
        total = sum(cnt.values())
        return cnt.get(word, 0) / total

    def predict_next(self, context_words: List[str], top_k=6, prefix: str = None) -> List[Tuple[str, float]]:
        """
        Predict next words given context_words (list). If prefix is provided,
        only return words starting with that prefix (useful for autocompletion).
        Returns list of (word, score) sorted descending.
        """
        tokens = [w.lower() for w in context_words if w]
        # candidate set: collect most frequent unigrams + any seen after contexts
        candidates = set()

        # add top unigrams as default candidates
        for w, _ in self.unigram_counts.most_common(200):
            candidates.add(w)

        # consider bigram and trigram continuations from context
        if len(tokens) >= 2:
            ctx_tri = tuple(tokens[-2:])
            trig_key = (2, ctx_tri)
            if trig_key in self.counts:
                candidates.update(self.counts[trig_key].keys())
        if len(tokens) >= 1:
            ctx_bi = tuple(tokens[-1:])
            bi_key = (1, ctx_bi)
            if bi_key in self.counts:
                candidates.update(self.counts[bi_key].keys())

        # scoring: interpolated probability
        scored = []
        for w in candidates:
            if prefix and not w.startswith(prefix.lower()):
                continue
            pu = self._prob_unigram(w)
            pb = 0.0
            pt = 0.0
            if len(tokens) >= 1:
                pb = self._prob_bigram(tuple(tokens[-1:]), w)
            if len(tokens) >= 2:
                pt = self._prob_trigram(tuple(tokens[-2:]), w)

            # interpolation: fallback gracefully if context not present
            score = (self.alpha_trigram * pt) + (self.alpha_bigram * pb) + (self.alpha_unigram * pu)
            scored.append((w, score))

        # if nothing matched prefix or candidates are empty, fallback to unigrams that match prefix
        if not scored and prefix:
            for w, cnt in self.unigram_counts.most_common(50):
                if w.startswith(prefix.lower()):
                    scored.append((w, cnt / self.total_unigrams))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def debug_context(self, context_words: List[str]):
        tokens = [w.lower() for w in context_words if w]
        print("Unigram top 10:", self.unigram_counts.most_common(10))
        if len(tokens) >= 1:
            print("Bigram counts for", tokens[-1:], self.counts.get((1, tuple(tokens[-1:])), None))
        if len(tokens) >= 2:
            print("Trigram counts for", tokens[-2:], self.counts.get((2, tuple(tokens[-2:])), None))

# quick demo
if __name__ == "__main__":
    with open("data/sample_corpus.txt", "r", encoding="utf8") as f:
        lines = [l.strip() for l in f if l.strip()]
    ngram = NGramPredictor(n=3)
    ngram.train(lines)
    print("Pred for 'i am':", ngram.predict_next(["i", "am"]))
    print("Pred for 'you are' with prefix 'd':", ngram.predict_next(["you", "are"], prefix="d"))
