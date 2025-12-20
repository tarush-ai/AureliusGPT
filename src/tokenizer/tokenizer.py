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
        self.d_model = d_model

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
        
        self.rules = []
        while len(vocab) < self.vocab_length:
            
            a = 0
            pairs = []

            while a < len(characters) - 1:
                if not characters[a+1] == "</w>":
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
                if i < len(characters) - 1 and characters[i] == rule[0][0] and characters[i+1] == rule[0][1]:
                    merged = rule[0][0] + rule[0][1]
                    characters2.append(merged)
                    i += 2
                else:    
                    characters2.append(characters[i])
                    i += 1


            characters = characters2

            self.rules.append((rule[0]))

        self.vocab = vocab
        
    def tokenize_train(self):
        self.bpe_train()
        tokens_to_ids = {}
        ids_to_tokens = {}
        for i in range(len(self.vocab)):
            tokens_to_ids[self.vocab[i]] =  i
            ids_to_tokens[i] = self.vocab[i]

        with open('data/vocabulary/tokens_to_ids.json', 'w') as file:
            json.dump(tokens_to_ids, file) 
        self.tokens_to_ids = tokens_to_ids

        with open('data/vocabulary/ids_to_tokens.json', 'w') as file:
            json.dump(ids_to_tokens, file)
        self.ids_to_tokens = ids_to_tokens

        if os.path.exists('data/weights/embeddings.npy'):
            E = np.load('data/weights/embeddings.npy')
        else:
            E = np.random.normal(0, 0.02, (self.vocab_length, self.d_model))
        self.E = E

        if not os.path.exists("data/vocabulary/embeddings.npy"):
            np.save("data/vocabulary/embeddings.npy", E)
        
        with open("data/vocabulary/rules.json", "w") as f:
            json.dump(self.rules, f)

        
    def encode(self, input_str):
        tokens = self.characterize(input_str, self.protected_tokens)

        i = 0
        
        for rule in self.rules:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == rule[0] and tokens[i+1] == rule[1]:
                    merged = tokens[i] + tokens[i+1]
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        

        return [self.tokens_to_ids[token] for token in tokens]


    def decode(self, input_nums):
        return [self.ids_to_tokens[int(i)] for i in input_nums]