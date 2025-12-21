import numpy as np
from tokenizer import Tokenizer

class Transformer:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.block1 = TransformerBlock()
        self.block2 = TransformerBlock()
        self.block3 = TransformerBlock()

    def forward(self, input_str):
        self.X = self.tokenizer.embed(input_str)
        


        



    def forward(self):



class TransformerBlock():



class AttentionBlock:

class LayerNorm:

class FFN: