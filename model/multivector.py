import time

import torch
import torch.nn as nn
from typing import Dict, List
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F


# 构建一个模型，不要BTN了
class PairwiseAttention(nn.Module):
    """
    Channels = 3, no additional linear transformation for now.
    Ref: Exploring Self-attention for Image Recognition
    """
    def __init__(self, h_dim: int, atn_type: str):
        super().__init__()
        self.atn_type = atn_type

        if self.atn_type in ['add', 'sub', 'Had']:
            self.fc = nn.Linear(h_dim, h_dim)
        elif self.atn_type == "concat":
            self.fc = nn.Linear(h_dim * 2, h_dim)
        elif self.atn_type == "cos":
            self.fc_add = nn.Linear(h_dim, h_dim)
            self.fc_hadamard = nn.Linear(h_dim, h_dim)
        else:
            raise ValueError(f"atn_type: {self.atn_type} not supported")

        self.norm1 = nn.LayerNorm(h_dim)
        self.norm2 = nn.LayerNorm(h_dim)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-2)  # Softmax along the channel dimension

    def forward(self, x):
        B, C, D = x.size()
        x_i = x.unsqueeze(2).expand(-1, -1, C, -1)
        x_j = x.unsqueeze(1).expand(-1, C, -1, -1)
        if self.atn_type == "cos":
            # Compute cosine similarity as gating coefficient
            cosine_similarity = F.cosine_similarity(x_i, x_j, dim=-1, eps=1e-8).unsqueeze(-1)
            # Ensure cosine similarity is between 0 and 1
            gate = (cosine_similarity + 1) / 2
            # Compute add and hadamard results
            add_result = self.fc_add((x_i + x_j) / 2)
            hadamard_result = self.fc_hadamard(x_i * x_j)
            # Combine results with gating
            relation = gate * add_result + (1 - gate) * hadamard_result
        elif self.atn_type == 'add':
            op_result = x_i + x_j
            relation = self.fc(op_result.view(B, C * C, -1)).view(B, C, C, D)
        elif self.atn_type == 'sub':
            op_result = x_i - x_j
            relation = self.fc(op_result.view(B, C * C, -1)).view(B, C, C, D)
        elif self.atn_type == 'Had':
            op_result = x_i * x_j
            relation = self.fc(op_result.view(B, C * C, -1)).view(B, C, C, D)
        elif self.atn_type == 'concat':
            op_result = torch.cat((x_i, x_j), dim=-1)
            relation = self.fc(op_result.view(B, C * C, -1)).view(B, C, C, D)

        weights = self.softmax(relation)

        # Computing weighted sum
        output = torch.einsum('bijc,bic->bjc', weights, x)
        output = self.norm1(output)  # Apply LayerNorm before adding residual
        output = self.activation(output)  # Apply ReLU activation
        output = output + x  # Residual connection
        output = self.norm2(output)  # Apply another LayerNorm after residual connection
        return output

class CosPairAttention(nn.Module):
    """
    Channels = 3, no additional linear transformation for now.
    Ref: Exploring Self-attention for Image Recognition
    """
    def __init__(self, modalities: List, h_dim: int, use_fusion_w: bool, use_fusion_shared: bool):
        super().__init__()
        self.modalities = modalities
        self.num_modality = len(self.modalities)
        self.h_dim = h_dim
        self.use_fusion_w = use_fusion_w
        self.use_fusion_shared = use_fusion_shared

        # 打印所有的设置
        print(f"modalities: {self.modalities}")
        print("use_fusion_w: ", self.use_fusion_w)
        print("use_fusion_shared: ", self.use_fusion_shared)

        # 根据模态个数配置多个MultiHeadAttention层, eg: Audio、Image、Text --> A2T, A2T, I2T
        self.proj_fc = nn.ModuleDict()
        if self.use_fusion_shared:  # 权重共享
            for i in range(self.num_modality):
                src_modal = self.modalities[i]
                for j in range(i + 1, self.num_modality):
                    tgt_modal = self.modalities[j]
                    self.proj_fc[f'{src_modal}2{tgt_modal}'] = nn.Linear(h_dim * 2, h_dim)
        else:
            for src in self.modalities:
                for tgt in self.modalities:
                    if src != tgt:
                        self.proj_fc[f'{src}2{tgt}'] = nn.Linear(h_dim * 2, h_dim)

        self.norm = nn.LayerNorm(h_dim)

    def forward(self, features):
        cross_relation = OrderedDict()
        new_features = {key: val.clone() for key, val in features.items()}

        cross_list = {k: [] for k in features.keys()}
        # 模态间进行Attention
        for src in self.modalities:
            for tgt in self.modalities:
                if src != tgt:
                    cross_list[src].append(tgt)

        for src, targets in cross_list.items():
            for tgt in targets:
                src_cat = torch.cat([features[src], features[tgt]], dim=-1)
                if f"{src}2{tgt}" in self.proj_fc:
                    k, v = self.proj_fc[f"{src}2{tgt}"](src_cat), self.proj_fc[f"{src}2{tgt}"](src_cat)
                    # cross_relation[f"{src}2{tgt}"] = torch.matmul(features[src], self.proj_fc[f"{src}2{tgt}"](
                    # src_cat).transpose(-2, -1))
                else:
                    k, v = self.proj_fc[f"{tgt}2{src}"](src_cat), self.proj_fc[f"{tgt}2{src}"](src_cat)
                    # cross_relation[f"{src}2{tgt}"] = torch.matmul(features[src],
                    #                                               .transpose(-2, -1)) / (self.h_dim ** 0.5)
                weight = torch.matmul(features[src], k.transpose(-2, -1)) / (self.h_dim ** 0.5)
                cross_relation[f"{src}2{tgt}"] = self.norm(torch.matmul(weight, v))
                # print(cross_relation[f"{src}2{tgt}"].shape)
                # exit()
        # Update features
        for src, targets in cross_list.items():
            A = features[src].clone()
            for tgt in targets:
                B = features[tgt].clone()
                if self.use_fusion_w:
                    cos_weight = F.cosine_similarity(A, B, dim=-1).unsqueeze(-1).detach()
                    new_features[src] += (cos_weight * cross_relation[f"{src}2{tgt}"])
                    print("cos_weight: ", cos_weight)
                else:
                    new_features[src] += (1 * cross_relation[f"{src}2{tgt}"])
            new_features[src] = self.norm(new_features[src])

        return new_features

class CosPairwiseAttention(nn.Module):
    """
    Channels = 3, no additional linear transformation for now.
    Ref: Exploring Self-attention for Image Recognition
    """
    def __init__(self, h_dim: int):
        super().__init__()
        self.fc = nn.Linear(h_dim * 2, h_dim)

        self.norm1 = nn.LayerNorm(h_dim)
        self.norm2 = nn.LayerNorm(h_dim)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-2)  # Softmax along the channel dimension

    def forward(self, x):
        B, C, D = x.size()
        x_i = x.unsqueeze(2).expand(-1, -1, C, -1)
        x_j = x.unsqueeze(1).expand(-1, C, -1, -1)
        cosine_similarity = F.cosine_similarity(x_i, x_j, dim=-1, eps=1e-8).unsqueeze(-1)

        # Compute the relationship between x_i and x_j
        op_result = torch.cat((x_i, x_j), dim=-1)
        relation = self.fc(op_result.view(B, C * C, -1)).view(B, C, C, D)

        # Multiply relation with cosine similarity
        relation = relation * cosine_similarity

        weights = self.softmax(relation)

        # Computing weighted sum
        output = torch.einsum('bijc,bic->bjc', weights, x)
        output = self.norm1(output)  # Apply LayerNorm before adding residual
        output = self.activation(output)  # Apply ReLU activation
        output = output + x  # Residual connection
        output = self.norm2(output)  # Apply another LayerNorm after residual connection
        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class Block(nn.Module):
    """
    Attention Block: (Alignment -->) Fusion
    """

    def __init__(self, modalities: List, h_dim: int, atn_type: str, use_align: bool = False,
                 use_fusion_w: bool = False, use_fusion_shared: bool = False):
        super().__init__()
        self.modalities = modalities
        self.h_dim = h_dim
        self.atn_type = atn_type
        self.align = use_align
        self.use_fusion_w = use_fusion_w
        self.use_fusion_shared = use_fusion_shared

        # Align
        if self.align:
            self.alignment = ChannelAttention(in_planes=2, ratio=2)
        # Fusion
        if self.atn_type == "cos":
            # self.fusion = CosPairwiseAttention(self.h_dim)
            self.fusion = CosPairAttention(self.modalities, self.h_dim, self.use_fusion_w, self.use_fusion_shared)
        else:
            self.fusion = PairwiseAttention(self.h_dim, self.atn_type)

    def forward(self, features, init_features):
        if self.align:
            aligned_features = []
            for name in features.keys():
                c_features = torch.cat([features[name], init_features[name]], dim=1)
                weights = self.alignment(c_features)
                aligned_features.append((weights * c_features).sum(dim=1, keepdim=True))
            features = OrderedDict({name: aligned_features[idx] for idx, name in enumerate(features.keys())})

        # Fusion: Concat as different channels
        # c_features = torch.cat(list(features.values()), dim=1)
        # print(c_features.shape)
        c_out = self.fusion(features)

        # 将输出的 c_out 对应到相应的 features 中
        # for idx, name in enumerate(features):
        #     features[name] = c_out[:, idx, :].unsqueeze(dim=1)
        return features, c_out


class MultiVector(nn.Module):
    def __init__(self, input_dims: Dict, hidden_dim: int, num_layers: int,
                 atn_type: str, vote: str, weights: List = None, use_align: bool = False, # use_fusion: bool = False,
                 use_fusion_w : bool = False,
                 use_fusion_shared: bool = False):
        super(MultiVector, self).__init__()
        self.input_dims = input_dims
        self.modalities = list(self.input_dims.keys())
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.atn_type = atn_type
        self.vote = vote
        self.weights = torch.Tensor(weights).cuda() if weights is not None else None
        self.use_align = use_align
        # self.use_fusion = use_fusion
        self.use_fusion_w = use_fusion_w
        self.use_fusion_shared = use_fusion_shared

        # Print the model configuration in a line
        print(f"Model Config:  --> {self.hidden_dim} --> {self.num_layers} --> {self.atn_type} --> {self.vote}")
        # other config
        print(f"align: {self.use_align}")
        # Project vectors
        self.proj = nn.ModuleDict({
            f"{k}_proj": nn.Sequential(
                nn.Linear(v, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            for k, v in input_dims.items()
        })

        # Attention Layer: (Alignment -->) Fusion
        self.atn_blocks = nn.ModuleDict({
            f"atn_block_{i}": Block(self.modalities, self.hidden_dim, self.atn_type, self.use_align, self.use_fusion_w, self.use_fusion_shared)
            for i in range(self.num_layers)
        })

        # Normalization Layer before classification
        self.norm = nn.LayerNorm(self.hidden_dim)

        # Classify Layer
        self.cls = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, **features):
        # Prepare the input -- numpy --> tensor
        for name, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[name] = torch.from_numpy(feature).type(torch.float32).to('cuda')

        # Project the vectors
        for name, feature in features.items():
            features[name] = self.proj[name + "_proj"](feature).unsqueeze(1)

        # Concat as different channels
        # Create the init features
        init_features = features.copy()
        for i in range(self.num_layers):
            features, c_features = self.atn_blocks[f"atn_block_{i}"](features, init_features)
        # print("Before CLS: c_feature: ", c_features.shape)

        # Feed to classifiear
        c_features = torch.cat(list(features.values()), dim=1)
        if self.vote == "avg":
            repr = torch.mean(c_features, dim=1)  # Average pooling


        # Normalize before classification
        out = self.cls(self.norm(repr))
        return out
