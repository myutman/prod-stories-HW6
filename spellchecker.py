import os
import re
import json
from collections import defaultdict

import numpy as np
import pandas as pd

import jellyfish

from fonetika.soundex import RussianSoundex
from fonetika.distance import PhoneticsInnerLanguageDistance

import hunspell


def text_cleaning(s):
    s = re.sub("^RT ", "", s)
    s = re.sub("@[\\w]*:?", "", s)
    s = re.sub("#[\\w]*", "", s)
    s = re.sub("https?://[\\w\./?=%]+", " ", s)
    s = re.sub("([XХ:, ]-?[ДD\)\(09]+|[оОoO0]_[оОoO0])", " ", s)
    s = re.sub("[\.,:?!\(\)—\-™…\"⚽✌️«»”/\\\\]+", " ", s)
    s = re.sub("&lt, 3", " ", s)
    s = re.sub("\*", "", s)
    s = s.lower()
    return ["<s>"] + s.split() + ["</s>"]


def calc_unigrams_and_bigrams(tokens, calc):
    for a, b in zip(tokens[:-1], tokens[1:]):
        calc[a + " " + b] += 1
    for a in tokens:
        calc[a] += 1
    return calc


def take_element(t):
    return "" if len(t) == 0 else t[0]


class SpellChecker:
    def fix_tokens(self, tokens):
        return [
            token if self.speller.spell(token)
            else take_element(self.speller.suggest(token))
        for token in tokens]

    
    def compute_bigram_probs(self):
        columns = ["id", "timestamp", "nickname", "text", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"]
        df = pd.read_csv("positive.csv", names=columns, sep=";")

        i = 0

        calc = defaultdict(int)
        n_tokens = 0
        tokenss = df[["text"]].applymap(text_cleaning).loc[np.random.choice(len(df), 1000)]
        tokenss = tokenss.applymap(self.fix_tokens)["text"]

        for tokens in tokenss:
            n_tokens += len(tokens)
            
            calc_tokens = calc_unigrams_and_bigrams(tokens, calc)

        return calc, n_tokens, len([key for key in calc.keys() if len(key.split()) == 1])
    
    
    def __init__(self):
        self.speller = hunspell.Hunspell("ru_RU")
        if not os.path.exists("precomputed.json"):
            self.calc, self.n_tokens, self.n_corpus_words = self.compute_bigram_probs()
            with open("precomputed.json", "w") as ouf:
                json.dump({"calc": self.calc, "n_tokens": self.n_tokens, "n_corpus_words": self.n_corpus_words}, ouf)
        else:
            with open("precomputed.json", "r") as inf:
                data = json.load(inf)
                self.calc = defaultdict(int, data["calc"])
                self.n_tokens = data["n_tokens"]
                self.n_corpus_words = data["n_corpus_words"]
                
        self.soundex = RussianSoundex(delete_first_letter=True)
        self.phon_distance = PhoneticsInnerLanguageDistance(self.soundex)
    
    
    def phon_sim(self, token, candidate):
        dist = self.phon_distance.distance(token, candidate)
        return 1 - dist / max(len(self.soundex.transform(token)), len(self.soundex.transform(candidate)))
    
    
    def get_prob(self, candidate):
        return np.log((self.calc[candidate] + 1) / (self.n_corpus_words + self.n_tokens))
    
    
    def get_forward_bigram_prob(self, token, candidate):
        return np.log((self.calc[token + " " + candidate] + 1) / (self.n_corpus_words + self.calc[token]))
    
    def get_backward_bigram_prob(self, candidate, token):
        return np.log((self.calc[candidate + " " + token] + 1) / (self.n_corpus_words + self.calc[token]))

    
    def suggest(self, s: str):
        tokens = text_cleaning(s)
        fixed_tokens = self.fix_tokens(tokens)
        for i, token in enumerate(tokens[1:-1], 1):
            if self.speller.spell(token):
                print(f"OK\t{token}")
            else:
                print(f"FIX\t{token}, candidates are:")
                candidates = self.speller.suggest(token)
                scores = []
                for candidate in candidates:
                    score = self.get_prob(candidate) \
                        + self.get_forward_bigram_prob(fixed_tokens[i - 1], candidate) \
                        + self.get_backward_bigram_prob(candidate, fixed_tokens[i + 1]) \
                        + self.phon_sim(token, candidate) \
                        + jellyfish.jaro_winkler(token, candidate)
                    scores.append(score)
                for candidate, score in sorted(list(zip(candidates, scores)), key=lambda x: -x[1]):
                    print(f"{candidate}: {score}")

    
    
