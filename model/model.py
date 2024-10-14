# model.py
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from typing import Dict
from collections import OrderedDict


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=512):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim)

    def forward(self, x):
        if self.training == True:
            chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)


class AbstractLayer(nn.Module):
    def __init__(self, base_input_dim: int, virtual_batch_size: int, bias: bool = True):
        """
        virtual_batch_size: int
        Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
        """
        super(AbstractLayer, self).__init__()
        self.fc = nn.Conv1d(base_input_dim, 2 * base_input_dim, kernel_size=1, bias=bias)
        initialize_glu(self.fc, input_dim=base_input_dim, output_dim=2 * base_input_dim)
        self.bn = GBN(2 * base_input_dim, virtual_batch_size)

    def forward(self, x, bottleneck):
        ori_dim = x.shape[1]
        btn_dim = bottleneck.shape[1]

        # concatenate bottleneck features with original features
        x = torch.cat([x, bottleneck], dim=1)  # [B, D] + [B, D'] -> [B, D + D']
        x = x.unsqueeze(-1)  # [B, D + D'] -> [B, D + D', 1]
        x = self.fc(x)  # [B, D + D', 1] -> [B, 2 * (D + D'), 1]
        x = self.bn(x)
        x = F.relu(torch.sigmoid(x[:, :ori_dim + btn_dim, :]) * x[:, ori_dim + btn_dim:, :])
        # split origin feature and bottleneck feature
        x, bottleneck = x[:, :ori_dim, :], x[:, ori_dim:, :]
        return x.squeeze(-1), bottleneck.squeeze(-1)


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


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """
    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


# Group Vector Attention Layer ref: https://github.dev/Pointcept/PointTransformerV2
class GroupedVectorAttention(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True
                 ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups)
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        query, key, value = self.linear_q(feat), self.linear_k(feat), self.linear_v(feat)
        key = pointops.grouping(reference_index, key, coord, with_xyz=True)
        value = pointops.grouping(reference_index, value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]
        relation_qk = key - query.unsqueeze(1)
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat



class our_model(nn.Module):
    def __init__(self, input_dims: Dict[str, int], num_bottleneck: int,
                 num_layers: int, virtual_batch_size: int, dropout_rate: float, repr_dim: int,
                 use_btn: bool = False
                 ):
        super(our_model, self).__init__()
        self.input_dims = input_dims
        self.virtual_batch_size = virtual_batch_size
        self.num_layers = num_layers
        self.use_btn = use_btn

        # First Step -- Generate Bottleneck Features
        self.init_btn = nn.Linear(sum(list(self.input_dims.values())), num_bottleneck)
        # 使用正态分布初始化权重，均值为0，标准差为0.02
        init.normal_(self.init_btn.weight, mean=0, std=0.02)
        init.constant_(self.init_btn.bias, 0)  # 初始化偏差为0

        # Second Step -- Align --> Fusion
        self.alignments = nn.ModuleDict({
            f"{name}_Align_{i}": ChannelAttention(in_planes=2, ratio=2)
            for i in range(self.num_layers)
            for name, dim in input_dims.items()
        })
        self.feature_tokenizers = nn.ModuleDict({
            f"{name}_ATN_{i}": AbstractLayer(base_input_dim=dim + num_bottleneck,
                                             virtual_batch_size=self.virtual_batch_size, bias=True)
            for i in range(self.num_layers)
            for name, dim in input_dims.items()
        })

        # Third Step -- Feed to representation layer
        self.repr_layers = self.output_layers = nn.ModuleDict({
            f"{name}_repr": nn.Sequential(
                nn.Linear(dim, repr_dim),
                nn.Dropout(dropout_rate),
            )
            for name, dim in input_dims.items()
        })

        # Forth Step -- Feed to the Output Layer
        # self.output_layers = nn.ModuleDict({
        #     f"{name}_fc": nn.Sequential(
        #         nn.LayerNorm(dim, eps=1e-5),
        #         nn.Dropout(dropout_rate),
        #         nn.Linear(dim, 1)
        #     )
        #     for name, dim in input_dims.items()
        # })
        # if self.use_btn:
        #     self.btn_fc = nn.Sequential(
        #         nn.LayerNorm(num_bottleneck, eps=1e-5),
        #         nn.GELU(),
        #         nn.Dropout(dropout_rate),
        #         nn.Linear(num_bottleneck, 1)
        #     )
        last_input_size = len(self.input_dims) * repr_dim
        if self.use_btn:
            last_input_size += num_bottleneck
        self.output = nn.Sequential(
            nn.LayerNorm(last_input_size, eps=1e-5),
            nn.ReLU(),
            nn.Linear(last_input_size, 1)
        )

    def forward(self, **features):
        # Prepare the input -- numpy --> tensor
        for name, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[name] = torch.from_numpy(feature).type(torch.float32).to('cuda')
        # Zero step -- Record Init features to Alignment
        init_features = features.copy()

        # First Step -- Generate Bottleneck Features
        x_btn = torch.cat([features[name] for name in self.input_dims.keys()], dim=1)
        bottleneck = self.init_btn(x_btn)

        # Second Step -- Concatenate Bottleneck Features with Original Features and Feed to the Abstract Layers
        for i in range(self.num_layers):
            btn_hats = OrderedDict()
            for name, feature in features.items():
                # alignment
                # 与对应的Init feature 拼接成 双通道的 feature
                c_features = torch.concat([feature.unsqueeze(1), init_features[name].unsqueeze(1)], dim=1)
                weights = self.alignments[f"{name}_Align_{i}"](c_features)
                feature = (weights * c_features).sum(dim=1, keepdim=True).squeeze()
                # fusion
                x, bottleneck = self.feature_tokenizers[f"{name}_ATN_{i}"](feature, bottleneck)
                btn_hats[name] = bottleneck
                features[name] = x
            # update the bottleneck features
            bottleneck = torch.stack([btn_hats[name] for name in self.input_dims.keys()], dim=-1)
            bottleneck = torch.mean(bottleneck, dim=-1)

        # Third Step -- Feed to the Output Layer
        # 1: Avg the prediction of each feature
        # out = {}
        # x_pool = 0.0
        # # Uni-modality
        # for name, feature in features.items():
        #     out[name] = self.output_layers[f"{name}_fc"](feature)
        #     x_pool += out[name]
        # if self.use_btn:
        #     # Bottleneck
        #     out['bottleneck'] = self.btn_fc(bottleneck)
        #     x_pool += self.btn_fc(bottleneck)
        # x_pool /= len(out)
        # return x_pool

        # try_3 先进行降维，再拼接
        repr_features = OrderedDict()
        for name, feature in features.items():
            repr_features[name] = self.repr_layers[f"{name}_repr"](feature)
        # 拼接
        if self.use_btn:
            tmp_feats = list(repr_features.values())
            tmp_feats.insert(-1, bottleneck)  # 三个特征，插入到中间
            all_feats = torch.cat(tmp_feats, dim=1)

        else:
            all_feats = torch.cat(list(repr_features.values()), dim=1)

        return self.output(all_feats)
