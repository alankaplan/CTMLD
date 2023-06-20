import pdb
import warnings
warnings.filterwarnings("error")
import numpy as np
import pickle

def randpos(M):
    return np.abs(np.random.randn(M, 1))

def norm_log(a):
    b = np.max(a, axis=0)[None, :]
    p = np.exp(a - b - np.log(np.sum(np.exp(a - b), axis=0))[None, :])
    return p

def log_sum_exp(a):
    b = np.max(a, axis=0)[None, :]
    p = b + np.log(np.sum(np.exp(a - b), axis=0))[None, :]
    return p

def likefun(alpha, lam, mu_X, sig_X, mu_Y, sig_Y, t, v_X, v_Y):
    a = np.log(alpha) + \
        np.log(lam) - \
        np.log(np.sqrt(2*np.pi*sig_X)) - np.log(np.sqrt(2*np.pi*sig_Y)) - \
        t*lam - \
        (mu_X - v_X)**2/(2*sig_X) - \
        (mu_Y - v_Y)**2/(2*sig_Y)
    return a

class model:
    def __init__(self, M, random_seed=24):
        np.random.seed(random_seed)
        self.alpha = 1/M*np.ones((M, 1))
        self.lam = randpos(M)
        self.mu_X = randpos(M)
        self.sig_X = randpos(M)
        self.mu_Y = randpos(M)
        self.sig_Y = randpos(M)

    def save(self, fname):
        with open(fname, 'wb') as fid:
            pickle.dump(self.__dict__, fid)

    def load(self, fname):
        with open(fname, 'rb') as fid:
            self.__dict__ = pickle.load(fid)

    def set_params(self, alpha, lam, mu_X, sig_X, mu_Y, sig_Y):
        self.alpha = alpha
        self.lam = lam
        self.mu_X = mu_X
        self.sig_X = sig_X
        self.mu_Y = mu_Y
        self.sig_Y = sig_Y

    def _expand(self, x):
        M = len(self.alpha)
        N = len(x[0])
        alpha = np.dot(self.alpha, np.ones((1, N)))
        lam = np.dot(self.lam, np.ones((1, N)))
        mu_X = np.dot(self.mu_X, np.ones((1, N)))
        sig_X = np.dot(self.sig_X, np.ones((1, N)))
        mu_Y = np.dot(self.mu_Y, np.ones((1, N)))
        sig_Y = np.dot(self.sig_Y, np.ones((1, N)))
        t = np.dot(np.ones((M, 1)), x[0][None, :])
        v_X = np.dot(np.ones((M, 1)), x[1][None, :])
        v_Y = np.dot(np.ones((M, 1)), x[2][None, :])

        return (alpha, lam, mu_X, sig_X, mu_Y, sig_Y, t, v_X, v_Y)

    def likelihood(self, x, takesum=True):
        (alpha, lam, mu_X, sig_X, mu_Y, sig_Y, t, v_X, v_Y) = self._expand(x)
        a = likefun(alpha, lam, mu_X, sig_X, mu_Y, sig_Y, t, v_X, v_Y)
        p = log_sum_exp(a)
        if takesum:
            return np.sum(p)
        else:
            return p

    def BIC(self, x, ll=None):
        M = len(self.alpha)
        N = len(x[0])
        num_params = (M - 1) + M*5
        if ll is None:
            ll = self.likelihood(x)
        bic = num_params*np.log(N) - 2*ll
        return bic

    def EMiter(self, x):
        # x is [t, v_X, v_Y]
        M = len(self.alpha)
        (alpha, lam, mu_X, sig_X, mu_Y, sig_Y, t, v_X, v_Y) = self._expand(x)

        p = self.Estep(alpha, lam, mu_X, sig_X, mu_Y, sig_Y, t, v_X, v_Y)
        (alpha, lam, mu_X, sig_X, mu_Y, sig_Y) = self.Mstep(p, t ,v_X, v_Y)
        ll = self.likelihood(x)
        new_model = model(M)
        new_model.set_params(alpha, lam, mu_X, sig_X, mu_Y, sig_Y)
        ll_new = new_model.likelihood(x)
        if ll_new > ll:
            self.set_params(alpha, lam, mu_X, sig_X, mu_Y, sig_Y)
            return ll_new
        return ll
         
    def Estep(self, alpha, lam, mu_X, sig_X, mu_Y, sig_Y, t, v_X, v_Y):
        a = likefun(alpha, lam, mu_X, sig_X, mu_Y, sig_Y, t, v_X, v_Y)
        p = norm_log(a)
        x = p + 1/p.shape[1]
        p = x/np.sum(x, axis=0)

        return p

    def Mstep(self, p, t, v_X, v_Y):
        N = t.shape[1]
        p_sum = np.sum(p, axis=1)

        alpha = (p_sum/N)[:, None]
        lam = (p_sum/np.sum(p*t, axis=1))[:, None]
        mu_X = (np.sum(p*v_X, axis=1)/p_sum)[:, None]
        sig_X = 0.01 + (np.sum(p*(v_X - mu_X)**2, axis=1)/p_sum)[:, None]
        mu_Y = (np.sum(p*v_Y, axis=1)/p_sum)[:, None]
        sig_Y = 0.01 + (np.sum(p*(v_Y - mu_Y)**2, axis=1)/p_sum)[:, None]

        return (alpha, lam, mu_X, sig_X, mu_Y, sig_Y)
