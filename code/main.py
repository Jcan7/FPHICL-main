# coding utf-8
import json, os
import random
import argparse
import copy
from generate_language import VocabHelp
import torch
import torch.nn.functional as F
from tqdm import trange
from data import load_data_instances,DataIterator
from contrastive_loss import HierarchicalContrastiveLoss
from model import MultiInferBert
import utils

def train(args):
    # load dataset
    train_sentence_packs = json.load(open(args.prefix + args.dataset + '/train.json'))  # 下载训练数据文件
    random.shuffle(train_sentence_packs)  # 将文件中的数据随机打乱
    dev_sentence_packs = json.load(open(args.prefix + args.dataset + '/dev.json'))  # 下载验证数据文件

    post_vocab = VocabHelp.load_vocab('data\D2/res16/vocab_post.vocab')
    deprel_vocab = VocabHelp.load_vocab('data\D2/res16/vocab_deprel.vocab')
    postag_vocab = VocabHelp.load_vocab('data\D2/res16/vocab_postag.vocab')
    synpost_vocab = VocabHelp.load_vocab('data\D2/res16/vocab_synpost.vocab')

    instances_train = load_data_instances(train_sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args)
    instances_dev = load_data_instances(dev_sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args)

    random.shuffle(instances_train)
    trainset = DataIterator(instances_train, args)
    devset = DataIterator(instances_dev, args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model = MultiInferBert(args).to(args.device)

    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': 5e-5},
        {'params': model.cls_linear.parameters()}
    ], lr=5e-5)

    best_joint_f1 = 0
    best_joint_epoch = 0
    for i in range(args.epochs):
        print('Epoch:{}'.format(i))
        for j in trange(trainset.batch_count):
            (bert_tokens, lengths, masks, sens_lens, token_ranges, aspect_tags, tags, new_tags, adjacency_matrix,
             sentiment_matrix, dep_matrix, pos_matrix, dis_matrix, tree_matrix, sentiment_label, aspect_label,
             opinion_label, sentiment_bert, aspect_bert, opinion_bert) = trainset.get_batch(j)

            preds = model(bert_tokens, sens_lens, adjacency_matrix, sentiment_matrix, dep_matrix, pos_matrix, dis_matrix
                          , tree_matrix, masks, sentiment_bert, aspect_bert, opinion_bert)

            aspect_label_normed, opinion_label_normed, sentiment_label_normed, final_pre = preds[0],preds[1], preds[2], preds[3]

            preds_flatten = final_pre.reshape([-1, final_pre.shape[3]])
            if args.tag_type == 'new':
                tags_flatten = new_tags.reshape([-1])
            else:
                tags_flatten = tags.reshape([-1])

            loss_fn = HierarchicalContrastiveLoss(temperature=0.7, loss_scaling_factor=0.1)
            loss_con = loss_fn(sentiment_label_normed, aspect_label_normed, opinion_label_normed, sentiment_label,
                               aspect_label, opinion_label)
            lp = F.cross_entropy(preds_flatten, tags_flatten, ignore_index=-1)
            loss = loss_con + lp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        joint_precision, joint_recall, joint_f1 = eval(model, devset, args)

        if joint_f1 > best_joint_f1:
            # model_path = args.model_dir + 'bert' + args.task + args.focus + '.pt'
            model_path = args.model_dir + 'bert' + args.task + '.pt'
            torch.save(model, model_path)
            best_joint_f1 = joint_f1
            best_joint_epoch = i
    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, args.task, best_joint_f1))


def eval(model, dataset, args):
    model.eval()
    with torch.no_grad():
        all_ids = []
        all_preds = []
        all_labels = []
        all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        all_pred_sent = []
        for i in range(dataset.batch_count):
            bert_tokens, lengths, masks, sens_lens, token_ranges, aspect_tags, tags, new_tags, adjacency_matrix, sentiment_matrix, dep_matrix, pos_matrix, dis_matrix, tree_matrix, sentiment_label, aspect_label, opinion_label, sentiment_bert, aspect_bert,opinion_bert = dataset.get_batch(i)

            preds = model(bert_tokens, sens_lens, adjacency_matrix, sentiment_matrix, dep_matrix, pos_matrix, dis_matrix
                          , tree_matrix, masks, sentiment_bert, aspect_bert, opinion_bert)[-1]
            if args.tag_type == 'new':
                if args.task == 'pair':
                    mask = torch.ByteTensor([[[[1, 0, 0, 1, 0, 0]]]]).cuda()
                else:

                    mask = torch.ByteTensor([[[[1, 1, 1, 1, 1, 1]]]]).cuda()
                pred_sent = copy.deepcopy(preds).masked_fill(mask==0, value=torch.tensor(-1e9))
                sent = torch.argmax(pred_sent, dim=3)
            else:
                sent = torch.argmax(preds, dim=3)
            preds = torch.argmax(preds, dim=3)
            all_preds.append(preds)
            all_pred_sent.append(sent)
            if args.tag_type == 'new':
                all_labels.append(new_tags)
            else:
                all_labels.append(tags)
            all_lengths.append(lengths)
            all_sens_lengths.extend(sens_lens)
            all_token_ranges.extend(token_ranges)

        all_pred_sent = torch.cat(all_pred_sent, dim=0).cpu().tolist()
        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

        metric = utils.Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1, sent=all_pred_sent)
        precision, recall, f1 = metric.score_uniontags()
        aspect_results = metric.score_aspect()
        opinion_results = metric.score_opinion()
        print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
                                                                  aspect_results[2]))
        print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
                                                                   opinion_results[2]))
        print(args.task + '\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

    model.train()
    return precision, recall, f1


def test(args):
    print("Evaluation on testset:")
    # model_path = args.model_dir + 'bert' + args.task + args.focus + '.pt'
    model_path = args.model_dir + 'bert' + args.task + '.pt'
    model = torch.load(model_path).to(args.device)
    model.eval()

    post_vocab = VocabHelp.load_vocab('data\D2/res16/vocab_post.vocab')
    deprel_vocab = VocabHelp.load_vocab('data\D2/res16/vocab_deprel.vocab')
    postag_vocab = VocabHelp.load_vocab('data\D2/res16/vocab_postag.vocab')
    synpost_vocab = VocabHelp.load_vocab('data\D2/res16/vocab_synpost.vocab')

    test_sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))

    ftest = args.prefix + args.dataset + '/test.json'

    instances = load_data_instances(test_sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args)
    testset = DataIterator(instances, args)
    eval(model, testset, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default="data\D2/",
                        help='dataset and embedding path prefix')
    # parser.add_argument('--prefix', type=str, default="../../data/ASTE_DATA_V2/",  # 调用add_argument方法调用对象
    #                     help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="./savemodel/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="triplet", choices=["pair", "triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--dataset', type=str, default="res16", choices=["res14", "lap14", "res15", "res16"],
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=100,
                        help='max length of a sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')

    parser.add_argument('--bert_model_path', type=str,
                        # default="pretrained/bert-base-uncased",
                        default="D:\ProjectJC/bert-base-uncased",
                        help='pretrained bert model path')
    parser.add_argument('--bert_tokenizer_path', type=str,
                        # default="pretrained/bert-base-uncased/bert-base-uncased-vocab.txt",
                        default="D:\ProjectJC/bert-base-uncased/vocab.txt",
                        help='pretrained bert tokenizer path')
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')
    parser.add_argument('--nhops', type=int, default=1,
                        help='inference times')
    parser.add_argument('--batch_size', type=int, default=32
                        ,
                        help='bathc size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=6,
                        help='label number')
    parser.add_argument("--SRD", default=5, type=int)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--pooling', default='avg', type=str, help='[max, avg, sum]')
    parser.add_argument('--gcn_dim', type=int, default=100, help='dimension of GCN')
    parser.add_argument('--current_run', type=int, default=0,
                        help='label number')
    parser.add_argument('--tag_type', type=str, default='new')
    args = parser.parse_args(args=[])

    if args.task == 'triplet':
        args.class_num = 6

    if args.mode == 'train':
        train(args)
        test(args)
    else:
        test(args)