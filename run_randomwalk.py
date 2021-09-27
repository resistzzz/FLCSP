import time
import datetime
import argparse

from data import *
from model import *
import os

parser = argparse.ArgumentParser(description='Mine potential topic distribution of items')
parser.add_argument('--data_path', default='data', help='dataset root path.')
parser.add_argument('--seed', type=int, default=1111, help='random seed for torch.')
parser.add_argument('--dataset', default='gowalla', help='dataset')
parser.add_argument('--anchor_num', type=int, default=400, help='number of potential topic for items.')
parser.add_argument('--N', type=int, default=1, help='number of heads for attention.')
parser.add_argument('--emb_size', type=int, default=50, help='dimension of item embeddings.')
parser.add_argument('--alpha', type=float, default=0.5, help='restart factor.')
parser.add_argument('--k', default=[20], help='top-k recommendation.')
parser.add_argument('--rw', default='w', help='read or write random walk result (r or w)')
parser.add_argument('--rw_path', default='rw', help='random walk result pickle')

args = parser.parse_args()

print(args)


def main(args):
    print('now dataset is ', args.dataset)
    data_path = os.path.join(args.data_path, args.dataset)
    corpus = Data(data_path)
    print('number of items: %d' % (corpus.dictionary.num_words-1))
    print('train_sess: %d' % len(corpus.train))
    print('test_sess: %d' % len(corpus.test))

    # ##############################################################
    rw_path = os.path.join(args.rw_path, args.dataset)
    rw_file = 'prob_n' + str(args.anchor_num) + '_restart' + str(args.alpha) + '.pk'
    rw_file = os.path.join(rw_path, rw_file)
    print('random walk file is: ', rw_file)
    if args.rw == 'w':
        t0 = time.time()
        Trans_adj, graph = build_graph(corpus.dictionary.num_words, corpus.train)
        anchors = anchor_select(graph, anchor_num=args.anchor_num)
        prob_conver = random_walk(Trans_adj, anchors, alpha=args.alpha)
        if os.path.exists(rw_path):
            pickle.dump(prob_conver, open(rw_file, 'wb'), protocol=4)
        else:
            os.mkdir(rw_path)
            pickle.dump(prob_conver, open(rw_file, 'wb'), protocol=4)
        print('random walk spend: %0.4f s' % (time.time() - t0))
    else:
        t0 = time.time()
        print('reading random walk result...')
        prob_conver = pickle.load(open(rw_file, 'rb'))
        print('reading random walk spend: %0.4f s' % (time.time() - t0))

if __name__ == '__main__':
    main(args)