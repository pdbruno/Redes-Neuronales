import numpy as np
from typing import Callable, List
import random


class PerceptronSimple:
    def __init__(self, N: int, M: int, g: Callable[[float], float], lr: float = 1e-5):
        self.g: Callable[[float], float] = g
        self.lr: float = lr
        self.w: np.ndarray = np.random.normal(0, 0.1, (N, M))

    def activacion(self, X_h: np.ndarray) -> np.ndarray:
        return self.g(np.dot(X_h, self.w))

    def estimacion(self, Z_h: np.ndarray, Y_h: np.ndarray) -> np.ndarray:
        return Z_h - Y_h

    def correccion(self, X_h: np.ndarray, E_h: np.ndarray) -> np.ndarray:
        return self.lr * np.outer(X_h, E_h)

    def fit(self, X: np.ndarray, Z: np.ndarray) -> List[float]:
        E = []
        e = 1
        t = 0
        while (e > 0.01) and (t < 999):
            e = 0
            numbers = list(range(X.shape[0]))
            random.shuffle(numbers)
            for h in numbers:
                Y_h = self.activacion(X[h]) #Y_h tiene que tener dimension (P, M)
                E_h = self.estimacion(Z[h], Y_h) #E_h tiene que tener dimension (P, M)
                delta_W = self.correccion(X[h], E_h)
                self.w += delta_W
                e += np.mean(np.square(E_h))
            t += 1
            E.append(e)
        return E
