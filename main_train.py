import time
import itertools
import os
from help import DataLoder, train, init_logger
from GTCN import GTCN


data_list = {
    'alpha_6_1_3': r'../data/alpha/alpha_20_sym_GTCN_6_1_3.mat',
}

lr = [0.05, 0.01, 0.005, 0.001]
embed = [10]
layer = [1, 2, 3]

for s in itertools.product(lr, embed, layer):
    setting = {
        'lr': s[0],
        'emb_size': s[1],
        'layers': s[2],
        'optim': 'Adam',
        'loss': 'huber',
    }

    for name in data_list.keys():
        data_name = name
        fname = data_list[name]

        logger = init_logger(data_name, s, fname)

        A, X, edges, target = DataLoder(fname, logger)

        # load model
        gcn = GTCN(A['train'], X['train'], edges['train'], fea_dim=setting['emb_size'], layers=setting['layers'])

        print(setting)
        logger.info(s)
        logger.info(setting)
        # train
        train(gcn, A, edges, target, logger, setting)

        print('train done!')
