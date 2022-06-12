import numpy as np


class AprendizajeHebbianoNoSupervisado:
    def __init__(self, N: int, M: int, regla: str, lr: float = 0.01, B: int = 1):
        self.regla = regla
        self.N: int = N
        self.M: int = M
        self.lr: float = lr
        self.w: np.ndarray = np.random.normal(-0.05, 0.05, (N, M))
        self.cached_identity = np.identity(M)
        self.cached_triu = np.triu(np.ones((M, M)))
        self.B: int = B

    def get_dW(self, Y, X):
        if self.regla == 'ojam':
            X_moño = np.dot(Y, self.w.T)
            return self.lr * np.outer(X-X_moño, Y)

        elif self.regla == 'sanger':
            D = self.cached_triu
            X_moño = np.dot(self.w, Y.T*D)
            return self.lr * (X.T - X_moño) * Y
        else:
            raise Exception('Regla no soportada')

    def fit(self, X: np.ndarray):
        o = 1
        P = X.shape[0]
        t = 0
        while o > 0.005:
            stochastic = np.random.permutation(P)
            for batch in range(0, P, self.B):
                h = stochastic[batch: batch + self.B]
                # no uso B porque quizas estoy en el ultimo step del range y el tamaño es < B
                batch_size = h.shape[0]
                Xh = X[h].reshape((batch_size, self.N))
                Y = np.dot(Xh, self.w)
                dW = self.get_dW(Y, Xh)
                self.w += dW
                o = np.sum(
                    np.abs(np.dot(self.w.T, self.w) - self.cached_identity))/2
            t+=1

""" 
class Caca(AprendizajeHebbianoNoSupervisado):
    def fit(self, X: np.ndarray):
        o = 1
        while o > 0.1:
            for x in X:
                Y = np.dot(x, self.w)
                dW = np.zeros((self.N, self.M))
                for j in range(self.M):
                    for i in range(self.N):
                        x_moño_i = sum([Y[k] * self.w[i, k] for k in range(j)])
                        dW = self.lr * (x[i] - x_moño_i) * Y[j]
                self.w += dW
        o = np.sum(
            np.abs(np.dot(self.w.T, self.w) - self.cached_identity))/2
 """