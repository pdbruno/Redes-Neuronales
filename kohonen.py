import numpy as num


class SelfOrganizingMap:
    def __init__(self, N: int, M: int, epochs: int, lr_st = 1.0, lr_dr = 0.1, ir_st = 3, ir_dr = 0.05):
        self.N: int = N
        self.M: int = M
        self.w: num.ndarray = num.random.normal(0, 1, (M, M, N))
        self.lr_st = lr_st
        self.lr_dr = lr_dr
        self.ir_st = ir_st
        self.ir_dr = ir_dr
        self.im = num.array([[(i,j) for j in range(M)] for i in range(M)])

    def fit(self, X: np.ndarray):
        o = 1
        P = X.shape[0]
        t = 0
        for t in range(epochs):
            stochastic = np.random.permutation(P)
            for i in range(0, P):
                h = stochastic[i]
                x = X[h].reshape((1, self.N))
                e = x-self.w
                n = num.linalg.norm(e, axis=2)
                p = num.unravel_index(num.argmin(n), n.shape)
                y = num.zeros((M, M))
                y[p] = 1
                
                lr = self.lr_st * num.exp( -t * self.lr_dr)
                
                ir = self.ir_st * num.exp( -t * self.ir_dr)

                d = num.linalg.norm( self.im-p, axis=2)
                pf = num.exp( -d / (2*num.square(ir))).reshape((M,M,1))

                dw = lr * pf * e
                self.w += dw
