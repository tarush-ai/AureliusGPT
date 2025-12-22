import numpy as np

class Util:
    def softmax(self, n):
        soft_n = []
        for i  in range(len(n)):
            row = n[i]
            soft_row = self.softmax_row(row)
            soft_n.append(soft_row)
        return np.array(soft_n)

    #softmax of an individual row, subtracting max
    def softmax_row(self, n):
        row = n
        maxval = np.max(row)
        exp = np.exp(row - maxval)
        soft_n = exp / np.sum(exp)
        return soft_n

    #relu
    def relu(self, n):
        relu_n = []
        for i in range(len(n)):
            n_row = []
            for j in range(len(n[i])):
                n_row.append(max(n[i][j], 0))
            relu_n.append(n_row)
        return np.array(relu_n)

    #derivative of relu for backpropogation
    def reluprime(self, n):
        relu_prime = np.zeros_like(n)
        for i in range(len(relu_prime)):
            for j in range(len(relu_prime[i])):
                if n[i][j] > 0: relu_prime[i][j] = 1
                else: relu_prime[i][j] = 0
        return relu_prime
