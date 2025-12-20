from collections import Counter
from .tokenizer import tokenize
import numpy as np
import re, json, os
from config import d_model, max_seq_length, vocab_length

class Tokenizer:
    def __init__(self):
        self.protected_tokens = set(["emphron", "sumphron", "huperphron", "nomos", "nemon", "axiopistos", "theophoretos", "oikonomian", "touto", "eferen", "auto", "eumoiros", "eudaimonia", "eupatridai", "kathoti", "katorthoseos", "kosmos", "melos", "meros", "pareilephamen", "symbainein", "tasis", "agathos", "aktines", "ekteinesthai", "daimon", "katorthoseis", "auto"])
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(PROJECT_ROOT, "data/processed/meditations.txt"), "r") as f:
            self.meditations = f.read()
        self.vocab_length = vocab_length

    def characterize(self, text, protected = None):
        newline = r"[\n]"        
        words = re.sub(newline, " ", text)
        words = words.split(" ")

        characters = []

        for word in words: 
            if protected and word not in protected:
                characters.extend([char for char in word])
            elif protected:
                characters.append(word)
            else:
                characters.extend([char for char in word])
            characters.append("</w>")            
        
        return characters
    
    def bpe_train(self):
        characters = self.characterize(self.meditations, self.protected_tokens)

        vocab = list(set(characters))
        vocab.extend(self.protected_tokens)


        while len(vocab) < self.vocab_length:
            
            a = 0
            pairs = []

            while a < len(characters) - 1:
                if a < characters[a:].index("</w>"):
                    pair = (characters[a], characters[a+1])
                    pairs.append(pair)
                    a += 1
                else:
                    a += 2

        
            rules = Counter(pairs).most_common()
            
            for i in rules:
                rulestr = ""
                for char in i[0]: rulestr += char
                if rulestr in vocab: continue
                else:
                    rule = i
                    vocab.append(rulestr)
                    break

            i = 0
            characters2 = []
            while i < len(characters):
                flag = False
                for j in range(len(rule)):
                    if i+j >= len(characters) or not characters[i+j] == rule[j]:
                        flag = True
                        characters2.append(characters[i])
                        i += 1
                        break
                if flag:
                    continue
                characters2.append(rulestr)
                i += len(rule)

            characters = characters2
        

    def tokenize():
