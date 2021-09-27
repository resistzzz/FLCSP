# -*- coding: utf-8 -*-

__author__ = 'zzz'

import argparse
import torch
import os
import time
import datetime

from data import *
from model import *

parser = argparse.ArgumentParser(description='Mine potential topic distribution of items')
parser.add_argument('--data_path', default='data', help='dataset root path.')
parser.add_argument('--seed', type=int, default=1111, help='random seed for torch.')
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of training epoch')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--l2', type=float, default=1e-5, help='l2-penalty')
parser.add_argument('--anchor_num', type=int, default=1000, help='number of potential topic for items.')
parser.add_argument('--hidden_size', type=int, default=50, help='gru hidden size.')
parser.add_argument('--nhid', type=int, default=50, help='causal conv hidden size.')
parser.add_argument('--hid_size', type=int, default=200, help='MLP hidden size for latent category')
parser.add_argument('--levels', type=int, default=1, help='number of layers')
parser.add_argument('--ksize', type=int, default=3, help='kernel size')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers')
parser.add_argument('--emb_dropout', type=float, default=0.2, help='dropout applied to embedding layer')
parser.add_argument('--N', type=int, default=1, help='number of heads for attention.')
parser.add_argument('--emb_size', type=int, default=50, help='dimension of item embeddings.')
parser.add_argument('--alpha', type=float, default=0.5, help='restart factor.')
parser.add_argument('--a', type=float, default=0.5, help='balance weight factor.')
parser.add_argument('--lr_dc', type=list, default=[10, 15], help='lr decay')
parser.add_argument('--lr_dc_cate', type=list, default=[10, 15], help='lr decay')
parser.add_argument('--k', default=[20], help='top-k recommendation.')
parser.add_argument('--clip', type=float, default=1.0, help='gradient clip (-1 means not clip)')
parser.add_argument('--rw', default='r', help='read or write random walk result (r or w)')
parser.add_argument('--rw_path', default='rw', help='random walk result pickle')
parser.add_argument('--model', default=False, help='Save model? True is, False not')
args = parser.parse_args()

print(args)
# torch.backends.cudnn.enabled = False
torch.manual_seed(args.seed)

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda: 0' if USE_CUDA else 'cpu')


def main(args):
    print('now dataset is ', args.dataset)
    data_path = os.path.join(args.data_path, args.dataset)
    corpus = Data(data_path)
    print('number of items: %d' % corpus.dictionary.num_words)

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

    t0 = time.time()
    train_data = batchify(corpus.train, args.batch_size)
    test_data = batchify(corpus.test, args.batch_size)
    print('process data: %0.4f s' % (time.time() - t0))

    num_chans = [args.nhid] * (args.levels - 1) + [args.emb_size]
    model_item = GRUTCN(corpus.dictionary.num_words, args.emb_size, args.hidden_size, num_chans, args.ksize,
                   args.dropout, args.emb_dropout, args, device=device)
    model_cate = MutiHeadAttn(corpus.dictionary.num_words, args.anchor_num, args.N, prob_conver, args, device=device)
    model = Fusion(corpus.dictionary.num_words, args, device=device)

    model_item = model_item.to(device)
    model_cate = model_cate.to(device)
    model = model.to(device)

    print('model_item: \n', model_item)
    print('-------------------------------------------')
    print('model_cate: \n', model_cate)
    print('-------------------------------------------')
    print('model_fusion: \n', model)

    epochs = args.epochs
    best_result = {}
    best_epoch = {}
    for k in args.k:
        best_result[k] = [0, 0]
        best_epoch[k] = [0, 0]
    t0 = time.time()
    for epoch in range(epochs):
        st = time.time()
        print('-------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(args, model_item, model_cate, model, train_data, test_data)
        for k in args.k:
            if hit[k] > best_result[k][0]:
                best_result[k][0] = hit[k]
                best_epoch[k][0] = epoch
                # if args.model:
                #     model_path = os.path.join('model', args.dataset)
                #     model_file = 'model_anchor_' + str(args.anchor_num) + '.pkl'
                #     model_path = os.path.join(model_path, model_file)
                #     torch.save(model.state_dict(), model_path)
            if mrr[k] > best_result[k][1]:
                best_result[k][1] = mrr[k]
                best_epoch[k][1] = epoch
            print('Hit@%d:\t%0.4f %%\tMRR@%d:\t%0.4f %%\t[%0.2f s]' % (k, hit[k], k, mrr[k], (time.time() - st)))

    print('------------------best result-------------------')
    for k in args.k:
        print('Best Result: Hit@%d: %0.4f %%\tMRR@%d: %0.4f %%\t[%0.2f s]' %
              (k, best_result[k][0], k, best_result[k][1], (time.time() - t0)))
        print('Best Epoch: Hit@%d: %d\tMRR@%d: %d\t[%0.2f s]' % (
            k, best_epoch[k][0], k, best_epoch[k][1], (time.time() - t0)))
    print('------------------------------------------------')
    print('Run time: %0.2f s' % (time.time() - t0))


def train_test(args, model_item, model_cate, model, tra_data, tes_data):
    print('start training: ', datetime.datetime.now())
    model_item.train()
    model_cate.train()
    model.train()
    model_item.scheduler.step()
    model_cate.scheduler.step()
    # model.scheduler.step()
    hit_dic_tra, mrr_dic_tra = {}, {}
    for k in args.k:
        hit_dic_tra[k] = []
        mrr_dic_tra[k] = []

    #####################################################################
    hit_dic_tra_item, mrr_dic_tra_item = {}, {}
    for k in args.k:
        hit_dic_tra_item[k] = []
        mrr_dic_tra_item[k] = []
    hit_dic_tra_cate, mrr_dic_tra_cate = {}, {}
    for k in args.k:
        hit_dic_tra_cate[k] = []
        mrr_dic_tra_cate[k] = []
    #####################################################################

    total_loss = []
    total_loss_item = []
    total_loss_cate = []
    n_batch = len(tra_data)
    for i in range(n_batch):
        # inp_front_tensor, lab_tensor = zero_padding_front(tra_data[i])
        inp_behind_tensor, lab_tensor = zero_padding_behind(tra_data[i])

        # inp_front_tensor = inp_front_tensor.to(device)
        inp_behind_tensor = inp_behind_tensor.to(device)
        lab_tensor = lab_tensor.to(device)

        model_item.optimizer.zero_grad()
        model_cate.optimizer.zero_grad()
        # model.optimizer.zero_grad()

        item_score = model_item(inp_behind_tensor)
        cate_score = model_cate(inp_behind_tensor)
        tra_score = model(item_score, cate_score)

        loss_item = model_item.loss_function(item_score, lab_tensor - 1)
        loss_cate = model_cate.loss_function(cate_score, lab_tensor - 1)
        loss = model.loss_function(tra_score, lab_tensor - 1)

        loss_item.backward()
        if args.clip > 0:
            gradClamp(model_item.parameters(), args.clip)
        model_item.optimizer.step()

        loss_cate.backward()
        if args.clip > 0:
            gradClamp(model_cate.parameters(), args.clip)
        model_cate.optimizer.step()

        # loss.backward()
        # if args.clip > 0:
        #     gradClamp(model.parameters(), args.clip)
        # model.optimizer.step()

        total_loss.append(loss.item())
        total_loss_item.append(loss_item.item())
        total_loss_cate.append(loss_cate.item())


        ##################################################################################
        ###
        # for k in args.k:
        #     predict = np.array(tra_score)
        #     for pred, target in zip(predict, lab_tensor.cpu().numpy()):
        #         hit_dic_tra[k].append(np.isin(target - 1, pred))
        #         if len(np.where(pred == target - 1)[0]) == 0:
        #             mrr_dic_tra[k].append(0)
        #         else:
        #             mrr_dic_tra[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))
        ###
        ##################################################################################


        for k in args.k:
            predict = tra_score.topk(k)[1]
            predict = predict.cpu()
            for pred, target in zip(predict, lab_tensor.cpu()):
                hit_dic_tra[k].append(np.isin(target - 1, pred))
                if len(np.where(pred == target - 1)[0]) == 0:
                    mrr_dic_tra[k].append(0)
                else:
                    mrr_dic_tra[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))

        ###############################################################################
        for k in args.k:
            predict = item_score.topk(k)[1]
            predict = predict.cpu()
            for pred, target in zip(predict, lab_tensor.cpu()):
                hit_dic_tra_item[k].append(np.isin(target - 1, pred))
                if len(np.where(pred == target - 1)[0]) == 0:
                    mrr_dic_tra_item[k].append(0)
                else:
                    mrr_dic_tra_item[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))
        for k in args.k:
            predict = cate_score.topk(k)[1]
            predict = predict.cpu()
            for pred, target in zip(predict, lab_tensor.cpu()):
                hit_dic_tra_cate[k].append(np.isin(target - 1, pred))
                if len(np.where(pred == target - 1)[0]) == 0:
                    mrr_dic_tra_cate[k].append(0)
                else:
                    mrr_dic_tra_cate[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))
        ###############################################################################

    print('Loss:\t%0.4f' % (np.mean(total_loss)))
    print('LossItem:\t%0.4f\tlr:\t%0.8f' % (np.mean(total_loss_item), model_item.optimizer.param_groups[0]['lr']))
    print('LossCate:\t%0.4f\tlr:\t%0.8f' % (np.mean(total_loss_cate), model_cate.optimizer.param_groups[0]['lr']))

    for k in args.k:
        hit_dic_tra[k] = np.mean(hit_dic_tra[k]) * 100
        mrr_dic_tra[k] = np.mean(mrr_dic_tra[k]) * 100
        hit_dic_tra_item[k] = np.mean(hit_dic_tra_item[k]) * 100
        mrr_dic_tra_item[k] = np.mean(mrr_dic_tra_item[k]) * 100
        hit_dic_tra_cate[k] = np.mean(hit_dic_tra_cate[k]) * 100
        mrr_dic_tra_cate[k] = np.mean(mrr_dic_tra_cate[k]) * 100
        print('Hit@%d:\t%0.4f %%\tMRR@%d:\t%0.4f %%' % (k, hit_dic_tra[k], k, mrr_dic_tra[k]))
        print('HitItem@%d:\t%0.4f %%\tMRRItem@%d:\t%0.4f %%' % (k, hit_dic_tra_item[k], k, mrr_dic_tra_item[k]))
        print('HitCate@%d:\t%0.4f %%\tMRRCate@%d:\t%0.4f %%' % (k, hit_dic_tra_cate[k], k, mrr_dic_tra_cate[k]))

    print('start predicting: ', datetime.datetime.now())
    model_item.eval()
    model_cate.eval()
    model.eval()
    hit_dic, mrr_dic = {}, {}
    for k in args.k:
        hit_dic[k] = []
        mrr_dic[k] = []
    ########################################################################
    hit_dic_item, mrr_dic_item = {}, {}
    for k in args.k:
        hit_dic_item[k] = []
        mrr_dic_item[k] = []
    hit_dic_cate, mrr_dic_cate = {}, {}
    for k in args.k:
        hit_dic_cate[k] = []
        mrr_dic_cate[k] = []
    ########################################################################
    total_loss = []
    total_loss_item = []
    total_loss_cate = []
    n_batch = len(tes_data)
    for i in range(n_batch):
        # inp_front_tensor_tes, lab_tensor_tes = zero_padding_front(tes_data[i])
        inp_behind_tensor_tes, lab_tensor_tes = zero_padding_behind(tes_data[i])
        # inp_front_tensor_tes = inp_front_tensor_tes.to(device)
        inp_behind_tensor_tes = inp_behind_tensor_tes.to(device)
        lab_tensor_tes = lab_tensor_tes.to(device)

        tes_score_item = model_item(inp_behind_tensor_tes)
        tes_score_cate = model_cate(inp_behind_tensor_tes)
        tes_score = model(tes_score_item, tes_score_cate)

        loss_item_tes = model_item.loss_function(tes_score_item, lab_tensor_tes - 1)
        loss_cate_tes = model_cate.loss_function(tes_score_cate, lab_tensor_tes - 1)
        loss_tes = model.loss_function(tes_score, lab_tensor_tes - 1)
        total_loss.append(loss_tes.item())
        total_loss_item.append(loss_item_tes.item())
        total_loss_cate.append(loss_cate_tes.item())

        ##################################################################################
        ###
        # for k in args.k:
        #     predict = np.array(tes_score)
        #     for pred, target in zip(predict, lab_tensor_tes.cpu().numpy()):
        #         hit_dic[k].append(np.isin(target - 1, pred))
        #         if len(np.where(pred == target - 1)[0]) == 0:
        #             mrr_dic[k].append(0)
        #         else:
        #             mrr_dic[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))
        # ###
        ##################################################################################

        for k in args.k:
            predict = tes_score.topk(k)[1]
            predict = predict.cpu()
            for pred, target in zip(predict, lab_tensor_tes.cpu()):
                hit_dic[k].append(np.isin(target - 1, pred))
                if len(np.where(pred == target - 1)[0]) == 0:
                    mrr_dic[k].append(0)
                else:
                    mrr_dic[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))

        ############################################################################
        for k in args.k:
            predict = tes_score_item.topk(k)[1]
            predict = predict.cpu()
            for pred, target in zip(predict, lab_tensor_tes.cpu()):
                hit_dic_item[k].append(np.isin(target - 1, pred))
                if len(np.where(pred == target - 1)[0]) == 0:
                    mrr_dic_item[k].append(0)
                else:
                    mrr_dic_item[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))
        for k in args.k:
            predict = tes_score_cate.topk(k)[1]
            predict = predict.cpu()
            for pred, target in zip(predict, lab_tensor_tes.cpu()):
                hit_dic_cate[k].append(np.isin(target - 1, pred))
                if len(np.where(pred == target - 1)[0]) == 0:
                    mrr_dic_cate[k].append(0)
                else:
                    mrr_dic_cate[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))
        ############################################################################
    print('Loss:\t%0.4f' % (np.mean(total_loss)))
    print('LossItem:\t%0.4f' % np.mean(total_loss_item))
    print('LossCate:\t%0.4f' % np.mean(total_loss_cate))

    for k in args.k:
        hit_dic[k] = np.mean(hit_dic[k]) * 100
        mrr_dic[k] = np.mean(mrr_dic[k]) * 100
        hit_dic_item[k] = np.mean(hit_dic_item[k]) * 100
        mrr_dic_item[k] = np.mean(mrr_dic_item[k]) * 100
        hit_dic_cate[k] = np.mean(hit_dic_cate[k]) * 100
        mrr_dic_cate[k] = np.mean(mrr_dic_cate[k]) * 100
        print('HitItem@%d:\t%0.4f %%\tMRRItem@%d:\t%0.4f %%' % (k, hit_dic_item[k], k, mrr_dic_item[k]))
        print('HitCate@%d:\t%0.4f %%\tMRRCate@%d:\t%0.4f %%' % (k, hit_dic_cate[k], k, mrr_dic_cate[k]))
    return hit_dic, mrr_dic


def gradClamp(parameters, clip=0.5):
    for p in parameters:
        p.grad.data.clamp_(min=-clip, max=clip)


main(args)
