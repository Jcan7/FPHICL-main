import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义图卷积层
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)  # 保证输出维度与输入一致

    def forward(self, x, adj):
        # 图卷积操作：H = A * X * W
        out = torch.matmul(adj, x)  # 使用邻接矩阵对输入特征进行加权求和
        out = self.fc(out)          # 通过全连接层进行线性变换
        return F.relu(out)          # 激活函数

# 定义特征提取模块
class FeatureExtractionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeatureExtractionModule, self).__init__()
        self.gcn = GCNLayer(input_dim, hidden_dim)  # 使用图卷积层
        self.fc = nn.Linear(hidden_dim, output_dim)  # 使用全连接层输出最终的特征

    def forward(self, feature_matrix, adj_matrix):
        # 输入到GCN进行特征提取
        gcn_output = self.gcn(feature_matrix, adj_matrix)
        return gcn_output  # 直接返回GCN的输出，而不是压缩为一个标量

# 定义多特征提取模型
class SynGCN(nn.Module):
    def __init__(self, bert_output_dim, gcn_hidden_dim):
        super(SynGCN, self).__init__()
        # 每个特征矩阵使用独立的GCN进行处理
        self.dep_module = FeatureExtractionModule(bert_output_dim, gcn_hidden_dim, bert_output_dim)  # 保持输出维度为bert_output_dim
        self.pos_module = FeatureExtractionModule(bert_output_dim, gcn_hidden_dim, bert_output_dim)
        self.dis_module = FeatureExtractionModule(bert_output_dim, gcn_hidden_dim, bert_output_dim)
        self.tree_module = FeatureExtractionModule(bert_output_dim, gcn_hidden_dim, bert_output_dim)

    def forward(self, dep_matrix, pos_matrix, dis_matrix, tree_matrix, adjacency_matrix):
        # 分别对每个矩阵进行GCN处理
        dep_features = self.dep_module(dep_matrix, adjacency_matrix)
        pos_features = self.pos_module(pos_matrix, adjacency_matrix)
        dis_features = self.dis_module(dis_matrix, adjacency_matrix)
        tree_features = self.tree_module(tree_matrix, adjacency_matrix)

        # 将每个模块的输出拼接
        # 拼接后的维度是 [batch_size, num_tokens, 4 * bert_output_dim]
        combined_features = torch.cat((dep_features, pos_features, dis_features, tree_features), dim=-1)

        return combined_features