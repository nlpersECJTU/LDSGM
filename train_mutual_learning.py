# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging as lgg
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam


def train(config, model, train_iter, dev_iter, test_iter):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters_rev = [
        {'params': [p for n, p in param_optimizer if any(rd in n for rd in ['decoder_reverse'])]}]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    optimizer_reverse = BertAdam(optimizer_grouped_parameters_rev,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 
    dev_best_acc_top = 0.0
    dev_best_acc_sec = 0.0
    dev_best_acc_conn = 0.0
    dev_best_f1_top = 0.0
    dev_best_f1_sec = 0.0
    dev_best_f1_conn = 0.0

    last_improve = 0  # 
    flag = False  # 
    model.train()
    criterion_kl_loss = nn.KLDivLoss(reduction='batchmean')
    for epoch in range(config.num_epochs):
        model.train()
        start_time = time.time()
        lgg.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, y1, y2, argmask) in enumerate(train_iter):

            outputs_top, outputs_sec, outputs_conn, outputs_top_reverse, outputs_sec_reverse, outputs_conn_reverse = model(trains, argmask)

            model.zero_grad()

            loss_top = F.cross_entropy(outputs_top, y1[0])
            loss_sec = F.cross_entropy(outputs_sec, y1[1])
            loss_conn = F.cross_entropy(outputs_conn, y1[2])

            loss_kl_top = criterion_kl_loss(torch.log_softmax(outputs_top, dim=-1),
                                            torch.softmax(outputs_top_reverse.detach(), dim=-1))
            loss_kl_sec = criterion_kl_loss(torch.log_softmax(outputs_sec, dim=-1),
                                            torch.softmax(outputs_sec_reverse.detach(), dim=-1))
            loss_kl_conn = criterion_kl_loss(torch.log_softmax(outputs_conn, dim=-1),
                                             torch.softmax(outputs_conn_reverse.detach(), dim=-1))
            # auxilary decoder loss
            loss_top_reverse = F.cross_entropy(outputs_top_reverse, y1[0])
            loss_sec_reverse = F.cross_entropy(outputs_sec_reverse, y1[1])
            loss_conn_reverse = F.cross_entropy(outputs_conn_reverse, y1[2])
            loss_kl_top_reverse = criterion_kl_loss(torch.log_softmax(outputs_top_reverse, dim=-1),
                                            torch.softmax(outputs_top.detach(), dim=-1))
            loss_kl_sec_reverse = criterion_kl_loss(torch.log_softmax(outputs_sec_reverse, dim=-1),
                                            torch.softmax(outputs_sec.detach(), dim=-1))
            loss_kl_conn_reverse = criterion_kl_loss(torch.log_softmax(outputs_conn_reverse, dim=-1),
                                             torch.softmax(outputs_conn.detach(), dim=-1))

            loss_kl = loss_kl_top + loss_kl_sec + loss_kl_conn
            loss_kl_reverse = loss_kl_top_reverse + loss_kl_sec_reverse + loss_kl_conn_reverse

            loss = loss_top * config.lambda1 + loss_sec * config.lambda2 + loss_conn * config.lambda3 + loss_kl* config.lambda4
            loss_reverse = loss_top_reverse * config.lambda1 + loss_sec_reverse * config.lambda2 + loss_conn_reverse * config.lambda3 + loss_kl_reverse*config.lambda4

            loss.backward(retain_graph=True)
            optimizer.step()
            loss_reverse.backward()
            optimizer_reverse.step()

            total_batch += 1
            if total_batch % 100 == 0:
                print(total_batch)

            if config.show_time:
                if total_batch % 100 == 0:
                    # 
                    y_true_top = y1[0].data.cpu()
                    y_true_sec = y1[1].data.cpu()
                    y_true_conn = y1[2].data.cpu()
                    if config.need_clc_loss:
                        # outputs_top_em = torch.div(outputs_top + outputs_top_reverse, 2)
                        # outputs_sec_em = torch.div(outputs_sec + outputs_sec_reverse, 2)
                        # outputs_conn_em = torch.div(outputs_conn + outputs_conn_reverse, 2)
                        y_predit_top = torch.max(outputs_top.data, 1)[1].cpu()
                        y_predit_sec = torch.max(outputs_sec.data, 1)[1].cpu()
                        y_predit_conn = torch.max(outputs_conn.data, 1)[1].cpu()

                        # y_predit_top_reverse = torch.max(outputs_top_reverse.data, 1)[1].cpu()
                        # y_predit_sec_reverse = torch.max(outputs_sec_reverse.data, 1)[1].cpu()
                        # y_predit_conn_reverse = torch.max(outputs_conn_reverse.data, 1)[1].cpu()
                    else:
                        y_predit_top = outputs_top.data.cpu()
                        y_predit_sec = outputs_sec.data.cpu()

                    train_acc_top = metrics.accuracy_score(y_true_top, y_predit_top)
                    train_acc_sec = metrics.accuracy_score(y_true_sec, y_predit_sec)
                    train_acc_conn = metrics.accuracy_score(y_true_conn, y_predit_conn)

                    # train_acc_top_reverse = metrics.accuracy_score(y_true_top, y_predit_top_reverse)
                    # train_acc_sec_reverse = metrics.accuracy_score(y_true_sec, y_predit_sec_reverse)
                    # train_acc_conn_reverse = metrics.accuracy_score(y_true_conn, y_predit_conn_reverse)

                    # train_acc_top_em = metrics.accuracy_score(y_true_top, y_predit_top)
                    # train_acc_sec_em = metrics.accuracy_score(y_true_sec, y_predit_sec)
                    # train_acc_conn_em = metrics.accuracy_score(y_true_conn, y_predit_conn)

                    loss_dev, acc_top, f1_top, acc_sec, f1_sec, acc_conn, f1_conn = evaluate(config, model, dev_iter)
                    # loss_dev_reverse, acc_top_reverse, f1_top_reverse, acc_sec_reverse, f1_sec_reverse,\
                    #     acc_conn_reverse, f1_conn_reverse = evaluate(config, model, dev_iter, reverse=True)
                    # loss_dev_em, acc_top_em, f1_top_em, acc_sec_em, f1_sec_em, acc_conn_em, f1_conn_em = evaluate(config, model, dev_iter, ensemble=True)

                    if (f1_top + f1_sec + f1_conn) > (dev_best_f1_top+ dev_best_f1_sec+ dev_best_f1_conn):
                        dev_best_f1_top = f1_top
                        dev_best_f1_sec = f1_sec
                        dev_best_f1_conn = f1_conn
                        dev_best_acc_top = acc_top
                        dev_best_acc_sec = acc_sec
                        dev_best_acc_conn = acc_conn
                        torch.save(model.state_dict(), config.save_path_top)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)

                    msg = 'top-down:TOP: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                          'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
                    lgg.info(msg.format(total_batch, loss.item(), train_acc_top, loss_dev, acc_top, f1_top, time_dif, improve))
                    msg = 'top-down:SEC: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                          'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
                    lgg.info(msg.format(total_batch, loss.item(), train_acc_sec, loss_dev, acc_sec, f1_sec, time_dif, improve))
                    msg = 'top-down:CONN: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                          'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
                    lgg.info(msg.format(total_batch, loss.item(), train_acc_conn, loss_dev, acc_conn, f1_conn, time_dif, improve))

                    lgg.info(' ')

                    # the auxilary decoder result
                    # msg = 'bottom-up:TOP Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                    #       'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
                    # lgg.info(msg.format(total_batch, loss_reverse.item(), train_acc_top_reverse, loss_dev_reverse, acc_top_reverse, f1_top_reverse, time_dif,
                    #                     ''))
                    # msg = 'bottom-up:SEC: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                    #       'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
                    # lgg.info(msg.format(total_batch, loss_reverse.item(), train_acc_sec_reverse, loss_dev_reverse, acc_sec_reverse, f1_sec_reverse, time_dif,
                    #                     ''))
                    # msg = 'bottom-up:CONN: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                    #       'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
                    # lgg.info(msg.format(total_batch, loss_reverse.item(), train_acc_conn_reverse, loss_dev_reverse, acc_conn_reverse, f1_conn_reverse, time_dif,
                    #                     ''))

                    # for the ensemble result
                    lgg.info(' ')
                    # msg = 'emsemble:TOP: Iter: {0:>6}, ' + \
                    #       'Val Loss: {1:>5.2},  Val Acc: {2:>6.2%}, Val F1: {3:>6.2%} Time: {4} {5}'
                    # lgg.info(msg.format(total_batch, loss_dev_em,
                    #                     acc_top_em, f1_top_em, time_dif,
                    #                     improve))
                    # msg = 'emsemble:SEC: Iter: {0:>6}, ' + \
                    #       'Val Loss: {1:>5.2},  Val Acc: {2:>6.2%}, Val F1: {3:>6.2%} Time: {4} {5}'
                    # lgg.info(msg.format(total_batch, loss_dev_em,
                    #                     acc_sec_em, f1_sec_em, time_dif,
                    #                     improve))
                    # msg = 'emsemble:CONN: Iter: {0:>6}, ' + \
                    #       'Val Loss: {1:>5.2},  Val Acc: {2:>6.2%}, Val F1: {3:>6.2%} Time: {4} {5}'
                    # lgg.info(msg.format(total_batch, loss_dev_em,
                    #                     acc_conn_em, f1_conn_em, time_dif,
                    #                     improve))
                model.train()

                if total_batch - last_improve > config.require_improvement:
                    # training stop
                    lgg.info("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
        if flag:
            break

        time_dif = get_time_dif(start_time)
        lgg.info("Train time usage: {}".format(time_dif))
        acc_top_test, f1_top_test, acc_sec_test, f1_sec_test, acc_conn_test, f1_conn_test \
            = test(config, model, test_iter)
        # acc_top_test_reverse, f1_top_test_reverse, acc_sec_test_reverse, f1_sec_test_reverse, acc_conn_test_reverse, f1_conn_test_reverse \
        #     = test(config, model, test_iter, reverse=True)
        # acc_top_test_reverse, f1_top_test_reverse, acc_sec_test_reverse, f1_sec_test_reverse, acc_conn_test_reverse, f1_conn_test_reverse \
        #     = test(config, model, test_iter, ensemble=True)
    dev_msg = 'dev_best_acc_top: {0:>6.2%},  dev_best_f1_top: {1:>6.2%}, \n' +\
                'dev_best_acc_sec: {2:>6.2%},  dev_best_f1_sec: {3:>6.2%}, \n' +\
                'dev_best_acc_conn: {4:>6.2%},  dev_best_f1_conn: {5:>6.2%}'
    lgg.info(dev_msg.format(dev_best_acc_top, dev_best_f1_top,
                            dev_best_acc_sec, dev_best_f1_sec,
                            dev_best_acc_conn, dev_best_f1_conn))


def test(config, model, test_iter,reverse=False, ensemble=False):
    model.load_state_dict(torch.load(config.save_path_top))
    model.eval()
    start_time = time.time()

    test_loss, acc_top, f1_top, report_top, confusion_top, \
        acc_sec, f1_sec, report_sec, confusion_sec, acc_conn, f1_conn = evaluate(config, model, test_iter, test=True, reverse=reverse, ensemble=ensemble)

    time_dif = get_time_dif(start_time)
    lgg.info("Test time usage: {}".format(time_dif))
    msg = 'TOP: Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test F1: {2:>6.2%}'
    lgg.info(msg.format(test_loss, acc_top, f1_top))
    msg = 'SEC: Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test F1: {2:>6.2%}'
    lgg.info(msg.format(test_loss, acc_sec, f1_sec))
    msg = 'CONN: Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test F1: {2:>6.2%}'
    lgg.info(msg.format(test_loss, acc_conn, f1_conn))
    lgg.info(report_top)
    lgg.info(report_sec)
    return acc_top, f1_top, acc_sec, f1_sec, acc_conn, f1_conn


def evaluate(config, model, data_iter, test=False, reverse=False, ensemble=False):
    model.eval()
    loss_total = 0
    predict_all_top = np.array([], dtype=int)
    labels1_all_top = np.array([], dtype=int)
    labels2_all_top = np.array([], dtype=int)

    predict_all_sec = np.array([], dtype=int)
    labels1_all_sec = np.array([], dtype=int)
    labels2_all_sec = np.array([], dtype=int)

    predict_all_conn = np.array([], dtype=int)
    labels1_all_conn = np.array([], dtype=int)
    labels2_all_conn = np.array([], dtype=int)

    with torch.no_grad():
        for texts, y1, y2, argmask in data_iter:

            outputs_top, outputs_sec, outputs_conn,outputs_top_reverse, outputs_sec_reverse, outputs_conn_reverse = model(texts, argmask)
            if reverse:
                outputs_top = outputs_top_reverse
                outputs_sec = outputs_sec_reverse
                outputs_conn = outputs_conn_reverse
            if ensemble:
                outputs_top = torch.div(outputs_top + outputs_top_reverse, 2)
                outputs_sec = torch.div(outputs_sec + outputs_sec_reverse, 2)
                outputs_conn = torch.div(outputs_conn + outputs_conn_reverse, 2)
            model.zero_grad()
            loss_top = F.cross_entropy(outputs_top, y1[0])
            loss_sec = F.cross_entropy(outputs_sec, y1[1])
            loss_conn = F.cross_entropy(outputs_conn, y1[2])

            loss = loss_top * config.lambda1 + loss_sec * config.lambda2 + loss_conn * config.lambda3

            loss_total += loss
            if config.need_clc_loss:
                y_predit_top = torch.max(outputs_top.data, 1)[1].cpu().numpy()
                y_predit_sec = torch.max(outputs_sec.data, 1)[1].cpu().numpy()
                y_predit_conn = torch.max(outputs_conn.data, 1)[1].cpu().numpy()

            y1_true_top = y1[0].data.cpu().numpy()
            y2_true_top = y2[0].data.cpu().numpy()
            labels1_all_top = np.append(labels1_all_top, y1_true_top)
            labels2_all_top = np.append(labels2_all_top, y2_true_top)
            predict_all_top = np.append(predict_all_top, y_predit_top)

            y1_true_sec = y1[1].data.cpu().numpy()
            y2_true_sec = y2[1].data.cpu().numpy()
            labels1_all_sec = np.append(labels1_all_sec, y1_true_sec)
            labels2_all_sec = np.append(labels2_all_sec, y2_true_sec)
            predict_all_sec = np.append(predict_all_sec, y_predit_sec)

            y1_true_conn = y1[2].data.cpu().numpy()
            y2_true_conn = y2[2].data.cpu().numpy()
            labels1_all_conn = np.append(labels1_all_conn, y1_true_conn)
            labels2_all_conn = np.append(labels2_all_conn, y2_true_conn)
            predict_all_conn = np.append(predict_all_conn, y_predit_conn)

    predict_sense_top = predict_all_top
    gold_sense_top = labels1_all_top
    mask = (predict_sense_top == labels2_all_top)
    gold_sense_top[mask] = labels2_all_top[mask]

    predict_sense_sec = predict_all_sec
    gold_sense_sec = labels1_all_sec
    mask = (predict_sense_sec == labels2_all_sec)
    gold_sense_sec[mask] = labels2_all_sec[mask]

    predict_sense_conn = predict_all_conn
    gold_sense_conn = labels1_all_conn
    mask = (predict_sense_conn == labels2_all_conn)
    gold_sense_conn[mask] = labels2_all_conn[mask]

    # cutoff
    if test:
        cut_off = 1039
    else:
        cut_off = 1165

    acc_top = metrics.accuracy_score(gold_sense_top, predict_sense_top)
    f1_top = metrics.f1_score(gold_sense_top, predict_sense_top, average='macro')

    gold_sense_sec = gold_sense_sec[: cut_off]
    predict_sense_sec = predict_sense_sec[: cut_off]
    acc_sec = metrics.accuracy_score(gold_sense_sec, predict_sense_sec)
    f1_sec = metrics.f1_score(gold_sense_sec, predict_sense_sec, average='macro')

    acc_conn = metrics.accuracy_score(gold_sense_conn, predict_sense_conn)
    f1_conn = metrics.f1_score(gold_sense_conn, predict_sense_conn, average='macro')

    if test:
        report_top = metrics.classification_report(gold_sense_top, predict_sense_top, target_names=config.i2top, digits=4)
        confusion_top = metrics.confusion_matrix(gold_sense_top, predict_sense_top)

        report_sec = metrics.classification_report(gold_sense_sec, predict_sense_sec, target_names=config.i2sec, digits=4)
        confusion_sec = metrics.confusion_matrix(gold_sense_sec, predict_sense_sec)

        return loss_total / len(data_iter), acc_top, f1_top, report_top, confusion_top, acc_sec, f1_sec, report_sec, confusion_sec, acc_conn, f1_conn
    return loss_total / len(data_iter), acc_top, f1_top, acc_sec, f1_sec, acc_conn, f1_conn

