import numpy as np
from typing import Callable, List
import random


class PerceptronMulticapa:
    def __init__(self, S: List[int], g: List[Callable[[float], float]], dgdx: List[Callable[[np.ndarray], np.ndarray]], lr: float = 0.1, B: int = 1):
        self.L: int = len(S)
        if(self.L != len(g)+1 and len(g) != len(dgdx)):
            raise Exception('Los tamaños de S, g o dgdx son incompatibles')

        self.g: List[Callable[[float], float]] = [lambda x: x, *g]
        self.dgdx: List[Callable[[np.ndarray], np.ndarray]] = [
            lambda x: 1, *dgdx]

        self.lr: float = lr

        self.W: List[ndarray] = [None if i == 0 else np.random.normal(
            0, 0.5, (S[i-1]+1, S[i])) for i in range(self.L)]

        self.B: int = B
        self.S: List[int] = S

    def bias_add(V):
        bias = -np.ones((len(V), 1))
        return np.concatenate((V, bias), axis=1)

    def bias_sub(V):
        return V[:, :-1]

    def activacion(self, Xh: np.ndarray) -> np.ndarray:
        Y = [np.zeros(shape=(Xh.shape[0], s + (i != self.L - 1)))
             for i, s in enumerate(self.S)]

        Y_moño = Xh

        for i in range(1, self.L):
            Y[i-1][:] = PerceptronMulticapa.bias_add(Y_moño)
            Y_moño = self.g[i](np.dot(Y[i-1], self.W[i]))

        Y[-1][:] = Y_moño
        return Y

    def correccion(self, Zh: np.ndarray, Y: np.ndarray) -> np.ndarray:

        dW = [None if i == 0 else np.zeros_like(
            w) for i, w in enumerate(self.W)]

        E = Zh - Y[-1]
        dY = self.dgdx[-1](Y[-1])
        D = E * dY

        for i in reversed(range(1, self.L)):
            dW[i] += self.lr * np.dot(Y[i-1].T, D)
            E = np.dot(D, self.W[i].T)
            dY = self.dgdx[i-1](Y[i-1])
            D = PerceptronMulticapa.bias_sub(E*dY)

        return dW

    def adaptacion(self, dW: List[np.ndarray]):
        for i in range(1, self.L):
            self.W[i] += dW[i]

    def fit(self, X: np.ndarray, Z: np.ndarray) -> List[float]:
        E = []
        e = 1
        t = 0
        P = X.shape[0]
        while (e > 0.01) and (t < 999):
            e = 0
            stochastic = np.random.permutation(P)

            for batch in range(0, P, self.B):
                h = stochastic[batch: batch + self.B]
                # no uso B porque quizas estoy en el ultimo step del range y el tamaño es < B
                batch_size = h.shape[0]
                # es importante que sean "vectores" columna (son matrices, no vectores pero se entiende la idea)
                Xh = X[h].reshape((batch_size, -1))
                Zh = Z[h].reshape((batch_size, -1))

                # devuelve las activaciones de todas las capas
                Yh = self.activacion(Xh)

                # devuelve las correcciones para todas las familias de pesos w
                dW = self.correccion(Zh, Yh)

                self.adaptacion(dW)

                e += np.mean(np.sum(np.square(Zh-Yh[-1]), axis=1))
            t += 1
            E.append(e)
        return E
