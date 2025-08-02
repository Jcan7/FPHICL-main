import math

import torch
import pickle
import numpy as np
import re
sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}
from transformers import BertTokenizer

def extract_entities(target_tags):
    # 使用正则表达式查找所有以 \B 开头的单词
    pattern = r'(\w+)\\B((?:\s+\w+\\I)*)'
    matches = re.finditer(pattern, target_tags)
    entity = ''
    # entities = []
    for match in matches:
        # 提取 \B 后的单词及其后续的 \I 标签单词
        entity_words = [match.group(1)]  # 第一个单词（\B 标签）

        # match.group(2) 包含所有后续的 "\I" 标签部分，例如 " York\I City\I"
        i_tags = match.group(2)

        if i_tags:
            # 使用正则表达式提取所有 \I 标签的单词
            subsequent = re.findall(r'\s+(\w+)\\I', i_tags)
            entity_words.extend(subsequent)

        # 将单词列表合并为一个字符串实体
        entity = ' '.join(entity_words)

    return entity

def get_sentiment_label(label_in):
    sentiment_dict = {
        'neutral': 0,
        'positive': 1,
        'negative': 2,
        'mixed': 3
    }
    label_2 = list(set([quad[2] for quad in label_in]))
    if len(label_2) == 1:
        label_2 = sentiment_dict[label_2[0]]
    else:
        label_2 = sentiment_dict['mixed']
    assert label_2 in [0, 1, 2, 3]

    return label_2

def get_aspect_labels(label_in):
    aspect_dict = {
        'short_aspect': 0,
        'long_aspect': 1,
        'mixed': 2,
    }
    label_2 = list(set([quad[0] for quad in label_in]))
    word_counts = [len(item.split()) for item in label_2]
    aspect_label_1 = 'unknown'
    if len(word_counts) == 1:
        if word_counts[0] == 1:
            aspect_label_1 = aspect_dict['short_aspect']
        else:
            aspect_label_1 = aspect_dict['long_aspect']
    elif len(word_counts) >= 2:
        if 1 in word_counts and any(count > 1 for count in word_counts):
            aspect_label_1 = aspect_dict['mixed']
        else:
            # 如果所有单词数都为1，则为 'short_aspect'
            if all(count == 1 for count in word_counts):
                aspect_label_1 = aspect_dict['short_aspect']
            elif all(count > 1 for count in word_counts):
                aspect_label_1 = aspect_dict['long_aspect']
            else:
                aspect_label_1 = aspect_dict['mixed']  # 处理其他可能的情况

    return aspect_label_1

def get_opinion_labels(label_in):
    opinion_dict = {
        'short_opinion': 0,
        'long_opinion': 1,
        'mixed': 2,
    }
    label_2 = list(set([quad[1] for quad in label_in]))
    word_counts = [len(item.split()) for item in label_2]
    opinion_label_1 = 'unknown'
    if len(word_counts) == 1:
        if word_counts[0] == 1:
            opinion_label_1 = opinion_dict['short_opinion']
        else:
            opinion_label_1 = opinion_dict['long_opinion']
    elif len(word_counts) >= 2:
        if 1 in word_counts and any(count > 1 for count in word_counts):
            opinion_label_1 = opinion_dict['mixed']
        else:

            if all(count == 1 for count in word_counts):
                opinion_label_1= opinion_dict['short_opinion']
            elif all(count > 1 for count in word_counts):
                opinion_label_1 = opinion_dict['long_opinion']
            else:
                opinion_label_1 = opinion_dict['mixed']  # 处理其他可能的情况

    return opinion_label_1


def get_spans(tags):
    '''for BIO tag'''
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def get_evaluate_spans(tags, length, token_range):
    '''for BIO tag'''
    spans = []
    start = -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l] == -1:
            continue
        elif tags[l] == 1:
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[l] == 0:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


class Instance(object):
    def __init__(self, tokenizer, sentence_pack, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.postag = sentence_pack['postag']
        self.head = sentence_pack['head']
        self.deprel = sentence_pack['deprel']

        self.tokens = self.sentence.strip().split()
        self.sen_length = len(self.tokens)
        self.token_range = []
        self.bert_tokens = tokenizer.encode(self.sentence)
        self.length = len(self.bert_tokens)
        self.bert_tokens_padding = torch.zeros(args.max_sequence_len).long()
        self.aspect_tags = torch.zeros(args.max_sequence_len).long()
        self.opinion_tags = torch.zeros(args.max_sequence_len).long()
        self.tags = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        self.new_tags = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        self.mask = torch.zeros(args.max_sequence_len)

        for i in range(self.length):
            self.bert_tokens_padding[i] = self.bert_tokens[i]
        self.mask[:self.length] = 1

        token_start = 1
        for i, w, in enumerate(self.tokens):
            token_end = token_start + len(tokenizer.encode(w, add_special_tokens=False))
            self.token_range.append([token_start, token_end-1])
            token_start = token_end
        assert self.length == self.token_range[-1][-1]+2

        self.aspect_tags[self.length:] = -1
        self.aspect_tags[0] = -1
        self.aspect_tags[self.length-1] = -1

        self.opinion_tags[self.length:] = -1
        self.opinion_tags[0] = -1
        self.opinion_tags[self.length - 1] = -1


        self.tags[:, :] = -1
        self.new_tags[:, :] = -1
        for i in range(1, self.length-1):
            for j in range(i, self.length-1):
                self.tags[i][j] = 0
                self.new_tags[i][j] = 0

        for triple in sentence_pack['triples']:
            aspect = triple['target_tags']
            opinion = triple['opinion_tags']
            aspect_span = get_spans(aspect)
            opinion_span = get_spans(opinion)

            '''set tag for aspect'''
            for l, r in aspect_span:#(l表示方面词的开始位置，r表示结束位置)
                start = self.token_range[l][0]
                end = self.token_range[r][0]#词向量后开始结束位置
                for i in range(start, end+1):
                    for j in range(i, i+1):
                        self.tags[i][j] = 1
                        self.new_tags[i][j] = 1
                        for q, k in aspect_span:
                            if l == q:
                                self.new_tags[i][j] = 1
                            else:
                                self.new_tags[i][j] = args.class_num - 2
                        if i < self.token_range[r][0]:
                            self.new_tags[i][j] = args.class_num - 2
                        else:
                            self.new_tags[i][j] = 1
                for i in range(l, r+1):
                    set_tag = 1 if i == l else 2
                    al, ar = self.token_range[i]#方面词中每个单词向量后的开始结束位置
                    self.aspect_tags[al] = set_tag
                    self.aspect_tags[al+1:ar+1] = -1
                    '''mask positions of sub words'''
                    self.tags[al+1:ar+1, :] = -1
                    self.tags[:, al+1:ar+1] = -1

            '''set tag for opinion'''
            for l, r in opinion_span:
                start = self.token_range[l][0]
                end = self.token_range[r][0]
                for i in range(start, end+1):
                    for j in range(i, i+1):
                        self.tags[i][j] = 2
                        self.new_tags[i][j] = 2
                        for q, k in opinion_span:
                            if l==q:
                                self.new_tags[i][j] = 2
                            else:
                                self.new_tags[i][j] = args.class_num - 1
                        if i < self.token_range[r][0]:
                            self.new_tags[i][j] = args.class_num - 1
                        else:
                            self.new_tags[i][j] = 2
                for i in range(l, r+1):
                    set_tag = 1 if i == l else 2
                    pl, pr = self.token_range[i]
                    self.opinion_tags[pl] = set_tag
                    self.opinion_tags[pl+1:pr+1] = -1
                    self.tags[pl+1:pr+1, :] = -1
                    self.tags[:, pl+1:pr+1] = -1

            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    sals = self.token_range[al][0]
                    sare = self.token_range[ar][0]
                    sprs, spre = self.token_range[pr]
                    sprs = self.token_range[pl][0]
                    spre = self.token_range[pr][0]
                    for k in range(sals, sare + 1):
                        for j in range(sprs,sprs+1):
                            if args.task == 'pair':
                                if k > j: self.new_tags[j][k] = 3
                                else: self.new_tags[k][j] = 3
                            else:
                                if k > j: self.new_tags[j][k] = sentiment2id[triple['sentiment']]
                                else: self.new_tags[k][j] = sentiment2id[triple['sentiment']]
                    for i in range(al, ar+1):
                        sal, sar = self.token_range[i]
                        for j in range(pl, pr+1):
                            # sal, sar = self.token_range[i]
                            spl, spr = self.token_range[j]
                            self.tags[sal:sar+1, spl:spr+1] = -1
                            if args.task == 'pair':
                                if i > j:
                                    self.tags[spl][sal] = 3
                                else:
                                    self.tags[sal][spl] = 3
                            elif args.task == 'triplet':
                                if i > j:
                                    self.tags[spl][sal] = sentiment2id[triple['sentiment']]
                                else:
                                    self.tags[sal][spl] = sentiment2id[triple['sentiment']]

        self.adjacency_matrix = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        for i in range(len(self.tokens)):
            start = self.token_range[i][0]
            end = self.token_range[i][1]
            for j in range(start, end + 1):
                s, e = self.token_range[self.head[i]-1]
                for k in range(s, e + 1):
                    self.adjacency_matrix[j][k] = 1
                    self.adjacency_matrix[k][j] = 1
                    self.adjacency_matrix[j][j] = 1

        self.sentiment_matrix = torch.zeros((args.max_sequence_len, args.max_sequence_len), dtype=torch.float32)
        for i in range(len(self.tokens)):
            start = self.token_range[i][0]
            end = self.token_range[i][1]
            for j in range(start, end + 1):
                s, e = self.token_range[self.head[i]-1]
                for k in range(s, e + 1):
                    self.sentiment_matrix[j][k] = self.head_score[i]
                    self.sentiment_matrix[k][j] = self.head_score[i]
                    self.sentiment_matrix[j][j] = self.self_score[i]

        self.pos_matrix = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        for i in range(len(self.tokens)):
            start, end = self.token_range[i][0], self.token_range[i][1]
            for j in range(len(self.tokens)):
                s, e = self.token_range[j][0], self.token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.pos_matrix[row][col] = postag_vocab.stoi.get(tuple(sorted([self.postag[i], self.postag[j]])))

        self.dep_matrix = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        for i in range(len(self.tokens)):
            start = self.token_range[i][0]
            end = self.token_range[i][1]
            for j in range(start, end + 1):
                s, e = self.token_range[self.head[i] - 1] if self.head[i] != 0 else (0, 0)
                for k in range(s, e + 1):
                    self.dep_matrix[j][k] = deprel_vocab.stoi.get(self.deprel[i])
                    self.dep_matrix[k][j] = deprel_vocab.stoi.get(self.deprel[i])
                    self.dep_matrix[j][j] = deprel_vocab.stoi.get('self')

        self.dis_matrix = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        for i in range(len(self.tokens)):
            start, end = self.token_range[i][0], self.token_range[i][1]
            for j in range(len(self.tokens)):
                s, e = self.token_range[j][0], self.token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.dis_matrix[row][col] = post_vocab.stoi.get(abs(row - col), post_vocab.unk_index)

        self.tree_matrix = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()

        # 初始化tmp为邻接矩阵
        tmp = [[0] * len(self.tokens) for _ in range(len(self.tokens))]
        for i in range(len(self.tokens)):
            j = self.head[i]
            if j != 0:
                tmp[i][j - 1] = tmp[j - 1][i] = 1

        # 计算词级度
        word_level_degree = [[4] * len(self.tokens) for _ in range(len(self.tokens))]
        for i in range(len(self.tokens)):
            node_set = {i}
            word_level_degree[i][i] = 0
            for j in tmp[i]:
                if j and j not in node_set:
                    word_level_degree[i][j] = 1
                    node_set.add(j)
                    for k in tmp[j]:
                        if k and k not in node_set:
                            word_level_degree[i][k] = 2
                            node_set.add(k)
                            for g in tmp[k]:
                                if g and g not in node_set:
                                    word_level_degree[i][g] = 3
                                    node_set.add(g)

        for i in range(len(self.tokens)):
            start, end = self.token_range[i]
            for j in range(len(self.tokens)):
                s, e = self.token_range[j]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.tree_matrix[row][col] = synpost_vocab.stoi.get(word_level_degree[i][j],
                                                                                  synpost_vocab.unk_index)

def load_data_instances(sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args):
    instances = list()
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)
    for sentence_pack in sentence_packs:
        instances.append(Instance(tokenizer, sentence_pack, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args))
    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)

    def get_batch(self, index):
        sentence_ids = []
        sentences = []
        sens_lens = []
        token_ranges = []
        bert_tokens = []
        lengths = []
        masks = []
        adjacency_matrix = []
        sentiment_matrix = []
        dep_matrix = []
        pos_matrix = []
        dis_matrix = []
        tree_matrix = []
        aspect_tags = []
        opinion_tags = []
        tags = []
        new_tags = []
        sentiment_label_out = []
        aspect_label_out = []
        opinion_label_out = []
        sentiment_label_bert = []
        aspect_label_bert = []
        opinion_label_bert = []

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            sentence_ids.append(self.instances[i].id)
            sentences.append(self.instances[i].sentence)
            sens_lens.append(self.instances[i].sen_length)
            token_ranges.append(self.instances[i].token_range)
            adjacency_matrix.append(self.instances[i].adjacency_matrix)
            sentiment_matrix.append(self.instances[i].sentiment_matrix)
            bert_tokens.append(self.instances[i].bert_tokens_padding)
            lengths.append(self.instances[i].length)
            masks.append(self.instances[i].mask)
            aspect_tags.append(self.instances[i].aspect_tags)
            opinion_tags.append(self.instances[i].opinion_tags)
            tags.append(self.instances[i].tags)
            new_tags.append(self.instances[i].new_tags)

            dep_matrix.append(self.instances[i].dep_matrix)
            pos_matrix.append(self.instances[i].pos_matrix)
            dis_matrix.append(self.instances[i].dis_matrix)
            tree_matrix.append(self.instances[i].tree_matrix)

            sentiment_label_out.append(self.instances[i].sentiment_label_in)
            aspect_label_out.append(self.instances[i].aspect_label_in)
            opinion_label_out.append(self.instances[i].opinion_label_in)

            sentiment_label_bert.append(self.instances[i].sentiment_label_bert)
            aspect_label_bert.append(self.instances[i].aspect_label_bert)
            opinion_label_bert.append(self.instances[i].opinion_label_bert)

        bert_tokens = torch.stack(bert_tokens).to(self.args.device)
        lengths = torch.tensor(lengths).to(self.args.device)
        masks = torch.stack(masks).to(self.args.device)
        adjacency_matrix = torch.stack(adjacency_matrix).to(self.args.device)
        sentiment_matrix = torch.stack(sentiment_matrix).to(self.args.device)
        aspect_tags = torch.stack(aspect_tags).to(self.args.device)
        opinion_tags = torch.stack(opinion_tags).to(self.args.device)
        tags = torch.stack(tags).to(self.args.device)
        new_tags = torch.stack(new_tags).to(self.args.device)
        dep_matrix = torch.stack(dep_matrix).to(self.args.device)
        pos_matrix = torch.stack(pos_matrix).to(self.args.device)
        dis_matrix = torch.stack(dis_matrix).to(self.args.device)
        tree_matrix = torch.stack(tree_matrix).to(self.args.device)

        sentiment_label_out = [torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in
                               sentiment_label_out]
        sentiment_label_out = torch.stack(sentiment_label_out).to(self.args.device)
        aspect_label_out = [torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in
                            aspect_label_out]
        aspect_label_out = torch.stack(aspect_label_out).to(self.args.device)
        opinion_label_out = [torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in
                             opinion_label_out]
        opinion_label_out = torch.stack(opinion_label_out).to(self.args.device)


        return (bert_tokens, lengths, masks, sens_lens, token_ranges, aspect_tags, tags, new_tags, adjacency_matrix, sentiment_matrix,
                dep_matrix, pos_matrix, dis_matrix, tree_matrix, sentiment_label_out, aspect_label_out, opinion_label_out, sentiment_label_bert, aspect_label_bert,
                opinion_label_bert)
