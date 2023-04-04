import torch
from torch import nn
import torch.nn.functional as F
import warnings
import math
from torch.nn.init import xavier_uniform_, constant_
from mmcv.ops import knn

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class DeformAttn(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_points=4):
        """
        Deformable Attention Module
        :param d_model      hidden dimension
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in DeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points

        # 每个query在每个特征层及每个注意力头都要采样n_points个点
        # 由于x，y，z坐标都有对应的偏移量，因此 * 3
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 3)
        # 每个query对应的所有采样点的注意力权重
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        # 线性变换得到value
        self.value_proj = nn.Linear(d_model, d_model)
        # 线性变换得到输出结果
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        constant_(self.sampling_offsets.bias.data, 0.)
        # dir_init = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
        # dir_init = torch.tensor(dir_init, dtype=torch.float32).view(self.n_heads, 1, 3).repeat(1, self.n_points, 1)
        # for i in range(self.n_points):
        #     dir_init[:, i, :] *= i + 1

        # with torch.no_grad():
        #     self.sampling_offsets.bias = nn.Parameter(dir_init.view(-1))
            
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, query_points, input, input_points, input_padding_mask=None):
        """
        :param query                       (B, Length_{query}, C)
        :param query_points                (B, Length_{query}, 3)
        :param input                       (B, Length_{input}, C)
        :param input_points                (B, Length_{input}, 3)
        :param input_padding_mask          (B, Length_{input}), True for padding elements, False for non-padding elements

        :return output                     (B, Length_{query}, C)
        """

        B, Len_q, _ = query.shape
        B, Len_in, _ = input.shape
        
        # 特征向量变换得到value, (B, Len_in, d_model=128)
        value = self.value_proj(input)
        # (B, Len_in, n_heads, d_model // n_heads)
        if input_padding_mask is not None:
            # 将原图padding的部分用0填充
            value = value.masked_fill(input_padding_mask[..., None], float(0))
            
        value = value.view(B, Len_in, self.n_heads, self.d_model // self.n_heads)
        # (B, Len_q, 8, 4, 3)
        sampling_offsets = self.sampling_offsets(query).view(B, Len_q, self.n_heads, self.n_points, 3)
        sampling_locations = query_points[:, :, None, None, :] + sampling_offsets
        # (B, Len_q, 8, 4) 预测采样点对应的注意力权重
        attention_weights = self.attention_weights(query).view(B, Len_q, self.n_heads, self.n_points)
        # (B, Len_q, 8, 4) 对4个点的权重进行归一化
        attention_weights = F.softmax(attention_weights, -1)
        # (B, Len_q, d_model)
        output = deform_attn(value, input_points, sampling_locations, attention_weights)
        # (B, Len_q, d_model)
        output = self.output_proj(output)
        return output


def deform_attn(value, value_points, sampling_points, attention_weights, k=1):
    """
    value: (B, Len_v, n_heads, d_model // n_heads)
    value_points: (B, Len_v, 3)
    sampling_points: (B, Len_q, n_heads, n_points, 3)
    attention_weights: (B, Len_q, n_heads, n_points)

    Return:
    output: (B, Len_q, C)
    """
    B_, Lv_, M_, D_ = value.shape
    _, Lq_, _, P_, _ = sampling_points.shape
    # (B_, Lq_*M_*P_, 3)
    sampling_points = sampling_points.view(B_, Lq_*M_*P_, 3)
    # (B_, Lq_*M_*P_, k)
    indices = knn(k, value_points, sampling_points).to(torch.int64).transpose(1, 2)
    # (B_*M_, Lq_*P_*k)
    indices = indices.reshape(B_, Lq_, M_, P_, k).transpose(1, 2).reshape(B_*M_, Lq_*P_*k)
    # (B_*M_, Lv_, D_)
    value = value.transpose(1, 2).flatten(0, 1)
    # (B_*M_, Lq_*P_*k, D_)
    features = torch.gather(value, dim=1, index=indices.unsqueeze(-1).repeat(1, 1, D_))
    # (B_*M_, Lq_*P_*k, D_) -> (B_*M_, Lq_, P_, k, D_) -> (B_*M_, Lq_, P_, D_) -> (B_*M_, D_, Lq_, P_)
    features = features.reshape(B_*M_, Lq_, P_, k, D_).sum(-2).permute(0, 3, 1, 2)
    # (B_, Lq_, M_, P_) -> (B_, M_, Lq_, P_) -> (B_*M_, 1, Lq_, P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(B_*M_, 1, Lq_, P_)
    # (B_*M_, D_, Lq_, P_) * (B_*M_, 1, Lq_, P_) = (B_*M_, D_, Lq_, P_)
    # -> (B_*M_, D_, Lq_) -> (B_, M_*D_, Lq_)
    features = (features * attention_weights).sum(-1).reshape(B_, M_*D_, Lq_)
    # (B_, Lq_, M_*D_)
    return features.transpose(1, 2).contiguous()

    
