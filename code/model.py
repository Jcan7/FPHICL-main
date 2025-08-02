import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sengcn import SenGCN
from syngcn import SynGCN
from torch.nn.functional import normalize
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
import torch
from new_gcn import GCNModel, GraphConvolution
import torch.nn as nn
from math import sqrt


class GatedFusion(nn.Module):
    def __init__(self, feature_dim=768):
        super(GatedFusion, self).__init__()
        # 用于计算门控权重的全连接层，输入为拼接后的特征，输出为一个权重向量
        self.gate_fc = nn.Linear(2 * feature_dim, feature_dim)
        # 另一条分支用全连接层对拼接后的特征进行映射
        self.fc = nn.Linear(2 * feature_dim, feature_dim)

    def forward(self, weighted_sen_gcn, weighted_syn_gcn):
        # 拼接两个特征 [32, 100, 100] -> [32, 100, 200]
        concatenated = torch.cat((weighted_sen_gcn, weighted_syn_gcn), dim=-1)
        # 计算门控权重，值域在 [0, 1]
        gate = torch.sigmoid(self.gate_fc(concatenated))  # [32, 100, 100]
        # 映射拼接后的特征
        fused = self.fc(concatenated)  # [32, 100, 100]
        # 采用门控机制融合：例如，用门控权重调控两个特征的贡献
        # 这里我们可以让融合特征为：门控 * fused + (1 - 门控) * (weighted_sen_gcn + weighted_syn_gcn)/2
        # 或者直接用门控对映射结果进行调节
        fused_features = gate * fused + (1 - gate) * ((weighted_sen_gcn + weighted_syn_gcn) * 0.5)
        return fused_features


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 计算 Q, K, V
        Q = self.q_linear(x)  # [batch_size, seq_len, embed_size]
        K = self.k_linear(x)  # [batch_size, seq_len, embed_size]
        V = self.v_linear(x)  # [batch_size, seq_len, embed_size]

        # 计算 attention scores
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attention_weights = F.softmax(attention_scores / (self.embed_size ** 0.5), dim=-1)  # 缩放

        # 计算 attention 输出
        attention_output = torch.bmm(attention_weights, V)  # [batch_size, seq_len, embed_size]

        # 输出线性变换
        output = self.out_linear(attention_output)  # [batch_size, seq_len, embed_size]

        return output


class CrossAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(CrossAttention, self).__init__()

        # 定义查询Q（BERT输出），键K和值V（GCN输出）的线性变换
        self.query_linear = nn.Linear(input_dim, attention_dim)
        self.key_linear = nn.Linear(input_dim, attention_dim)
        self.value_linear = nn.Linear(input_dim, attention_dim)

        # 通过一个线性层来组合注意力后的输出
        self.output_linear = nn.Linear(attention_dim, input_dim)

    def forward(self, bert_output, gcn_output):
        """
        bert_output: BERT的输出 [batch_size, seq_len, input_dim]
        gcn_output: GCN的输出 [batch_size, num_nodes, input_dim]
        """
        # 查询Q为BERT的输出
        Q = self.query_linear(bert_output)  # [batch_size, seq_len, attention_dim]

        # 键K和值V为GCN的输出
        K = self.key_linear(gcn_output)  # [batch_size, num_nodes, attention_dim]
        V = self.value_linear(gcn_output)  # [batch_size, num_nodes, attention_dim]

        # 计算注意力得分，Q * K^T / sqrt(d_k)
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # [batch_size, seq_len, num_nodes]
        attention_scores = attention_scores / (K.size(-1) ** 0.5)  # 缩放

        # 通过Softmax获取权重
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, seq_len, num_nodes]

        # 加权平均值
        attention_output = torch.bmm(attention_weights, V)  # [batch_size, seq_len, attention_dim]

        # 通过一个线性层组合输出
        output = self.output_linear(attention_output)  # [batch_size, seq_len, input_dim]

        return output


class MultiInferBert(torch.nn.Module):
    def __init__(self, args):
        super(MultiInferBert, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)

        # self.triplet_biaffine = Biaffine(args, args.gcn_dim, args.gcn_dim, args.class_num, bias=(True, True))
        self.ap_fc = nn.Linear(args.bert_feature_dim, args.gcn_dim)
        self.op_fc = nn.Linear(args.bert_feature_dim, args.gcn_dim)
        self.dim_per_head = args.bert_feature_dim
        self.sqrt_dim = math.sqrt(self.dim_per_head)
        self.num_layers = args.num_layers
        self.dense = nn.Linear(args.bert_feature_dim, args.gcn_dim)
        self.gcn_layers1 = nn.ModuleList()
        self.gcn_layers2 = nn.ModuleList()

        self.linear_q = torch.nn.ModuleList([torch.nn.Linear(768, 64).to('cuda') for _ in range(args.class_num)])
        self.linear_k = torch.nn.ModuleList([torch.nn.Linear(768, 64).to('cuda') for _ in range(args.class_num)])
        self.linear_v = torch.nn.ModuleList([torch.nn.Linear(100, 64).to('cuda') for _ in range(args.class_num)])

        self.relu = torch.nn.ReLU()

        self.linear1 = nn.Linear(2 * args.bert_feature_dim, args.bert_feature_dim)

        # self.cll_linear = torch.nn.Linear(args.bert_feature_dim, args.class_num)
        self.dropout_output = torch.nn.Dropout(0.1)

        self.gcn = SenGCN(args.bert_feature_dim, 128, 100)
        self.syn_gcn = SynGCN(args.gcn_dim, 128)
        self.cross_attention = CrossAttention(768, 384)

        self.cls_linear = torch.nn.Linear(2 * args.class_num, args.class_num)

        self.conv = nn.Conv2d(6, 1, kernel_size=1)
        self.fusion_model = GatedFusion(feature_dim=768)
        self.gcn_layer = GCNModel(args)
        self.W_1 = torch.nn.Parameter(torch.randn(100, 768))
        self.W_2 = torch.nn.Parameter(torch.randn(100, 768))
        self.W = torch.nn.Parameter(torch.randn(768, 768))

        self.dropout1 = torch.nn.Dropout(p=0.3)
        self.dropout2 = torch.nn.Dropout(p=0.3)

        self.self_attention = SelfAttention(embed_size=768)

        self.q = torch.nn.Linear(args.bert_feature_dim, 384)
        self.k = torch.nn.Linear(args.bert_feature_dim, 384)
        self.v = torch.nn.Linear(args.bert_feature_dim, 384)

    def _lstm_feature(self, embedding, lengths):
        embedding = pack_padded_sequence(embedding, lengths.cpu(), batch_first=True)
        context, _ = self.bilstm(embedding)
        context, _ = pad_packed_sequence(context, batch_first=True)
        return context

    def BiAffine(self, H1, W, H2):
        H_out = torch.matmul(H1, W)
        H_out = torch.matmul(H_out, H2.permute(0, 2, 1))
        H_out = torch.matmul(F.softmax(H_out, dim=-1), H1)
        return H_out

