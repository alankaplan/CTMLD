from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import numpy as np
import model
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

def conv_test(v, thresh=0.01, nt=10):
    if len(v) > nt:
        v = np.array(v)
        d = -1*np.diff(v)/v[0:-1]
        if np.all(d[-1*nt:] < thresh):
            return False
    return True

def train_model(M, x, random_seed=24):
    N = len(x[0])
    m = model.model(M, random_seed=random_seed)
    b = []
    cont = True
    while cont:
        ll = m.EMiter(x)
        b.append(m.BIC(x, ll)/N)
        sys.stdout.write('    ll: %f, bic: %f\n'%(ll, b[-1]))
        sys.stdout.flush()
        cont = conv_test(b)
    bic = b[-1]
    sys.stdout.write('%i: %.06f\n'%(M, bic))
    sys.stdout.flush()
    return (m, bic)

def train_model_bbfunc(M, x, random_seed=24):
    N = len(x[0])
    m = model.model(M, random_seed=random_seed)
    b = []
    cont = True
    sys.stdout.write('M=%i\n'%M)
    sys.stdout.flush()
    while cont:
        ll = m.EMiter(x)
        b.append(m.BIC(x, ll)/N)
        sys.stdout.write('    ll: %f, bic: %f\n'%(ll, b[-1]))
        sys.stdout.flush()
        cont = conv_test(b)
    bic = b[-1]
    sys.stdout.write('%i: %.06f\n'%(M, bic))
    sys.stdout.flush()
    return -1*bic

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dat_file', action='store', help='data file')
    parser.add_argument('log_file', action='store', help='output json log')
    parser.add_argument('model_file', action='store', help='output model')
    parser.add_argument('-m', action='store', type=int, dest='num_comp', default=None, help='number of components, leave out to perform model selection')
    parser.add_argument('-n', action='store', type=int, dest='num_samp', default=None, help='number of samples to train on, leave out to use all')
    parser.add_argument('-s', action='store', type=int, dest='rand_seed', default=24, help='random seed for EM initialization')

    args = parser.parse_args()
    
    x = np.loadtxt(args.dat_file)

    if args.num_samp is not None:
        x = x[:, 0:args.num_samp]

    print(len(x[0]))

    if args.num_comp is None:
        pbounds = {'M': (1, 1000)}
        bbfunc = lambda M : train_model_bbfunc(int(M), x, random_seed=args.rand_seed)
        bounds_transformer = SequentialDomainReductionTransformer()
        optimizer = BayesianOptimization(f=bbfunc, pbounds=pbounds, random_state=1, bounds_transformer=bounds_transformer)
        logger = JSONLogger(path=args.log_file)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(init_points=4, n_iter=20)
        M = optimizer.max['params']['M']
    else:
        M = args.num_comp

    (m, bic) = train_model(int(M), x, random_seed=args.rand_seed)

    with open(args.model_file, 'wb') as fid:
        pickle.dump(m, fid)
