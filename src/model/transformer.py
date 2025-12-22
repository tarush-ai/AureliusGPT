import concurrent.futures
import numpy as np
from tokenizer.tokenizer import Tokenizer
from config import PROJECT_ROOT, d_k, d_v, d_model, d_ff, h, num_blocks, epsilon
from util import Util
import os

class Transformer:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.relpath = os.path.join(PROJECT_ROOT, "data", "weights")
        self.blocks = [TransformerBlock(num, self.relpath) for num in range(num_blocks)]

    def forward(self, input_str):
        self.encoded = self.tokenizer.encode(input_str)
        self.X = self.tokenizer.embed(self.encoded) + self.tokenizer.position(self.encoded)

        self.Y = self.blocks[0].forward(self.X)

        for pos in range(len(self.blocks)-1):
            self.Y = self.blocks[pos+1].forward(self.Y)
        
        return self.Y

    def ce_loss(predicted, actual):
        return -np.log(predicted * actual)
    
class TransformerBlock:
    def __init__(self, num, path):
        block = "block" + str(num)
        self.relpath = os.path.join(path, block)
        self.attentionblock = AttentionBlock(self.relpath)
        self.normone = LayerNorm(1, self.relpath)
        self.ffn = FFN(self.relpath)
        self.normtwo = LayerNorm(2, self.relpath)

    def forward(self, X):
        MHA = self.attentionblock.forward(X)
        AddNorm = self.norm.forward(MHA, X)
        FFN = self.ffn.forward(AddNorm)
        output = self.norm.forward(FFN, X)
        return output

class AttentionBlock:
    def __init__(self, path):
        self.relpath = os.path.join(path, "attention")

        self.heads = [AttentionHead() for i in range(h)]

        self.d_head = d_model / h

        self.Wq_path = os.path.join(self.relpath, "Wq.npy")
        self.Wk_path = os.path.join(self.relpath, "Wk.npy")
        self.Wv_path = os.path.join(self.relpath, "Wv.npy")
        self.Wo_path = os.path.join(self.relpath,  "Wo.npy")
        
        if os.path.exists(self.Wq_path):
            self.Wq = np.load(self.Wq_path)
        else:
            self.Wq = np.random.normal(0, 0.02, (d_model, d_k))
            np.save(self.Wq_path, self.Wq)
        
        if os.path.exists(self.Wk_path):
            self.Wk = np.load(self.Wk_path)
        else:
            self.Wk = np.random.normal(0, 0.02, (d_model, d_k))
            np.save(self.Wk_path, self.Wk)
        
        if os.path.exists(self.Wv_path):
            self.Wv = np.load(self.Wv_path)
        else:
            self.Wv = np.random.normal(0, 0.02, (d_model, d_v))
            np.save(self.Wv_path, self.Wv)
        
        if os.path.exists(self.Wo_path):
            self.Wo = np.load(self.Wo_path)
        else:
            self.Wo = np.random.normal(0, 0.02, (d_model, d_model))
            np.save(self.Wo_path, self.Wo)

    def forward(self, X):
        
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
    
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i, head in enumerate(self.heads):
                startindex = i * self.d_head
                endindex = (i+1) * self.d_head - 1 
                Qh = Q[startindex:endindex]
                Kh = K[startindex:endindex]
                Vh = V[startindex:endindex]

                future = executor.submit(head.forward, Qh, Kh, Vh)
                futures.append(future)

        MHA = np.zeros(len(X), d_model)

        for future in futures:
            MHA.extend(future.result()) #is this correct? idk if this works for np arrays
        
        MHA = MHA @ self.Wo
        return MHA
            
class LayerNorm:
    def __init__(self, num, path):
        numpath = "norm" + str(num)
        self.relpath = os.path.join(path, numpath)
        self.gamma_path = os.path.join(self.relpath, "gamma.npy")
        self.beta_path = os.path.join(self.relpath, "beta.npy")

        if os.path.exists(self.gamma_path):
            self.gamma = np.load(self.gamma_path)
        else:
            self.gamma = np.random() #is this the right approach?
            np.save(self.gamma_path, self.gamma)
        if os.path.exists(self.beta_path):
            self.beta = np.load(self.beta_path)
        else:
            self.beta = np.random()
            np.save(self.beta_path, self.beta)
    
    def forward(self, pred, X):
        X = pred + X
        lnorm = np.zeros_like(pred)
        for xt in X:
            mean = np.mean(X)
            stddev = np.std(X)
            norm = self.gamma * (xt - mean) / np.sqrt(stddev ** 2 + epsilon) + self.beta
            lnorm.append(norm)
        return lnorm
        
class FFN:
    def __init__(self, path):
        self.relpath = os.path.join(path, "ffn")

        self.util = Util()

        self.W1path = os.path.join(self.relpath, "W1.npy")
        self.b1path = os.path.join(self.relpath, "b1.npy")
        self.W2path = os.path.join(self.relpath, "W2.npy")
        self.b2path = os.path.join(self.relpath, "b2.npy")

        if os.path.exists(self.W1path):
            self.W1 = np.load(self.W1path)
        else:
            self.W1 = np.random.normal(0, 0.02, (d_model, d_ff))
        if os.path.exists(self.b1path):
            self.b1 = np.load(self.b1path)
        else:
            self.b1 = np.random.normal(0, 0.02, (d_ff,))
        if os.path.exists(self.W2path):
            self.W2 = np.load(self.W2path)
        else:
            self.W2 = np.random.normal(0, 0.02, (d_ff, d_model))
        if os.path.exists(self.b2path):
            self.b2 = np.load(self.b2path)
        else:
            self.b2 = np.random.normal(0, 0.02, (d_model,))


    def forward(self, X):
        h1 = X @ self.W1 + self.b1
        h2 = self.util.relu(h1)
        h3 = h2 @ self.W2 + self.b2
        return h3

class AttentionHead:
    def __init__(self):
        self.util = Util()
    
    def forward(self, Qh, Kh, Vh):
        scores = Qh @ Kh.T
        scores += self.util.mask(len(scores), len(scores[0]))
        d_head = d_model / h
        scores /= np.sqrt(d_head)

        attentionh = self.util.softmax(scores) @ Vh
        return attentionh    
