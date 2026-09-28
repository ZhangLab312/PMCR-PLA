import torch
import numpy as np
from sympy.integrals.risch import residue_reduce
from torch import nn
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"
conv_filters = [[1, 32], [3, 32], [5, 64], [7, 128]]
embedding_size = output_dim = 256
d_ff = 256
n_heads = 8
d_k = 16
n_layer = 1
MODEL_DROPOUT = 0.2

smi_vocab_size = 53
seq_vocab_size = 21

seed = 3407

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
np.random.seed(seed)


class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class ConvEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, conv_filters, output_dim, type):
        super().__init__()
        if type == 'seq':
            self.embed = nn.Embedding(vocab_size, embedding_size)
        elif type == 'poc':
            self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        self.convolutions = nn.ModuleList()
        for kernel_size, out_channels in conv_filters:
            conv = nn.Conv1d(embedding_size, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
            self.convolutions.append(conv)
        self.num_filters = sum([f[1] for f in conv_filters])
        self.projection = nn.Linear(self.num_filters, output_dim)

    def forward(self, inputs):
        embeds = self.embed(inputs).transpose(-1, -2)
        conv_hidden = []
        for layer in self.convolutions:
            conv = F.relu(layer(embeds))
            conv_hidden.append(conv)
        res_embed = torch.cat(conv_hidden, dim=1).transpose(-1, -2)
        res_embed = self.projection(res_embed)
        return res_embed


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(embedding_size, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(embedding_size, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(embedding_size, d_k * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_k, embedding_size, bias=False)
        self.ln = nn.LayerNorm(embedding_size)
        self.attn = SelfAttention()

    def forward(self, input_Q, input_K, input_V, attn_mask):
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context = self.attn(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_k)
        output = self.fc(context)
        return self.ln(output)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, embedding_size, bias=False)
        )
        self.ln = nn.LayerNorm(embedding_size)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.ln(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi = MultiHeadAttention()
        self.feed = FeedForward()

    def forward(self, en_input, attn_mask):
        context = self.multi(en_input, en_input, en_input, attn_mask)
        output = self.feed(context + en_input)
        return output


class Seq_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_emb = ConvEmbedding(seq_vocab_size, embedding_size, conv_filters, output_dim, 'seq')
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layer)])

    def forward(self, seq_input):
        output_emb = self.seq_emb(seq_input)
        enc_self_attn_mask = get_attn_pad_mask(seq_input, seq_input)
        for layer in self.layers:
            output_emb = layer(output_emb, enc_self_attn_mask)
        return output_emb


class Smi_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(smi_vocab_size, embedding_size, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embedding_size)
        self.self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(embedding_size, n_heads, d_ff, dropout=MODEL_DROPOUT)
            for _ in range(n_layer)
        ])
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, smi_input):
        x = self.embedding(smi_input)
        x = self.positional_encoding(x)
        attn_mask = get_attn_pad_mask(smi_input, smi_input)
        for layer in self.self_attention_layers:
            x = layer(x, attn_mask)
        return self.layer_norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=MODEL_DROPOUT):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.feed_forward = FeedForward()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        attn_output = self.self_attn(x, x, x, attn_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x


class MultiScaleCrossAttention(nn.Module):
    def __init__(self, feature_dim=256, kernel_sizes=[1, 3, 5], num_heads=8):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.feature_dim = feature_dim

        self.seq_convs = nn.ModuleList()
        self.smi_convs = nn.ModuleList()

        for k in kernel_sizes:
            seq_conv = nn.Conv1d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=k,
                padding=k // 2,
                groups=feature_dim
            )
            self.seq_convs.append(seq_conv)

            smi_conv = nn.Conv1d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=k,
                padding=k // 2,
                groups=feature_dim
            )
            self.smi_convs.append(smi_conv)

        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                batch_first=True
            ) for _ in range(len(kernel_sizes))
        ])

        self.fusion_layer = nn.Linear(len(kernel_sizes) * feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.gelu = nn.GELU()

    def forward(self, seq_features, smi_features, seq_mask=None, smi_mask=None):
        multi_scale_outputs = []

        for i, (seq_conv, smi_conv, cross_attn) in enumerate(zip(
                self.seq_convs, self.smi_convs, self.cross_attentions)):
            seq_conv_input = seq_features.transpose(1, 2)
            seq_conv_output = seq_conv(seq_conv_input)
            seq_conv_output = seq_conv_output.transpose(1, 2)

            smi_conv_input = smi_features.transpose(1, 2)
            smi_conv_output = smi_conv(smi_conv_input)
            smi_conv_output = smi_conv_output.transpose(1, 2)

            cross_attn_output, _ = cross_attn(
                query=seq_conv_output,
                key=smi_conv_output,
                value=smi_conv_output,
                key_padding_mask=smi_mask
            )

            multi_scale_outputs.append(cross_attn_output)

        if len(multi_scale_outputs) > 1:
            concatenated_features = torch.cat(multi_scale_outputs, dim=-1)
        else:
            concatenated_features = multi_scale_outputs[0]

        fused_features = self.fusion_layer(concatenated_features)
        fused_features = self.layer_norm(fused_features)
        fused_features = self.gelu(fused_features)

        return fused_features


class ImprovedCrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim=256, correlation_dim=64, kernel_sizes=[1, 3, 5, 7]):
        super().__init__()
        self.feature_dim = feature_dim
        self.correlation_dim = correlation_dim

        self.multi_scale_cross_attn = MultiScaleCrossAttention(
            feature_dim=feature_dim,
            kernel_sizes=kernel_sizes
        )

        self.self_attention_blocks = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            ) for _ in range(2)
        ])


        self.final_compress = nn.Linear(feature_dim, correlation_dim)

        self.fusion_layer = nn.Linear(2 * feature_dim + correlation_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.gelu = nn.GELU()

    def forward(self, seq_features, smi_features, seq_mask=None, smi_mask=None):

        cross_features = self.multi_scale_cross_attn(seq_features, smi_features, seq_mask, smi_mask)


        attended_features = cross_features


        correlation_features = self.final_compress(attended_features)
        correlation_features = self.gelu(correlation_features)


        def masked_mean(features, mask):
            valid = (~mask).unsqueeze(-1).to(features.dtype)
            return (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


        global_seq = masked_mean(seq_features, seq_mask)
        global_smi = masked_mean(smi_features, smi_mask)
        global_features = masked_mean(correlation_features, seq_mask)

        stacked_features = torch.cat([global_seq, global_smi, global_features], dim=-1)

        fused_features = self.fusion_layer(stacked_features)
        fused_features = self.layer_norm(fused_features)
        fused_features = self.gelu(fused_features)

        return fused_features


class PocketPositionLoss(nn.Module):


    def __init__(self, position_weight=1.0, length_weight=1.0, coverage_weight=0.4, size_penalty=0.5):
        super().__init__()
        self.position_weight = position_weight
        self.length_weight = length_weight
        self.coverage_weight = coverage_weight
        self.size_penalty = size_penalty


        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')

    def forward(self, pred_start, pred_end, true_start, true_end):

        start_loss = self.smooth_l1(pred_start, true_start)
        end_loss = self.smooth_l1(pred_end, true_end)


        pred_length = pred_end - pred_start
        true_length = true_end - true_start


        length_error = torch.abs((pred_length - true_length) / (true_length + 1e-7))
        length_loss = torch.mean(length_error)


        overlap_start = torch.max(pred_start, true_start)
        overlap_end = torch.min(pred_end, true_end)
        overlap_length = torch.clamp(overlap_end - overlap_start, min=0)

        coverage_ratio = overlap_length / (true_length + 1e-7)
        coverage_loss = torch.mean(1.0 - coverage_ratio)


        size_ratio = pred_length / (true_length + 1e-7)

        size_error = torch.abs(size_ratio - 1.0)
        size_penalty_loss = torch.mean(size_error)


        total_loss = (self.position_weight * (start_loss + end_loss) +
                      self.length_weight * length_loss +
                      self.coverage_weight * coverage_loss +
                      self.size_penalty * size_penalty_loss)


        return total_loss, start_loss, end_loss, length_loss, coverage_loss, size_penalty_loss

class MyModule(nn.Module):
    def __init__(self, max_seq_len=1024):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.seq_encoder = Seq_Encoder()
        self.smi_encoder = Smi_Encoder()


        self.caff_fusion = ImprovedCrossAttentionFusion(
            feature_dim=embedding_size,
            correlation_dim=64,
            kernel_sizes=[1, 3, 5, 7]
        )

        # 亲和力预测分支
        self.affinity_fc = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            Squeeze()
        )

        # 口袋位置预测分支 - 使用全局特征预测起始和结束位置
        self.pocket_position_fc = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.Dropout(MODEL_DROPOUT),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(MODEL_DROPOUT),
            nn.PReLU(),
            nn.Linear(64, 2),  # 输出两个值：起始位置和结束位置
            nn.Sigmoid()  # 使用Sigmoid将输出限制在0-1范围（归一化位置）
        )

        # 口袋位置损失函数
        self.pocket_loss_fn = PocketPositionLoss(position_weight=1.0, length_weight=1.0, coverage_weight=0.4, size_penalty=0.5)

    def forward(self, seq_encode, smi_encode):
        # 编码蛋白质序列
        seq_outputs = self.seq_encoder(seq_encode)  # [batch_size, seq_len, embedding_size]
        # 编码配体
        smi_outputs = self.smi_encoder(smi_encode)

        # 创建mask
        seq_mask = (seq_encode == 0)
        smi_mask = (smi_encode == 0)

        # 关键修改：交叉注意力融合返回两个输出
        global_features = self.caff_fusion(seq_outputs, smi_outputs, seq_mask, smi_mask)

        # 预测亲和力 - 使用全局特征
        affinity = self.affinity_fc(global_features)

        # 预测口袋起始和结束位置
        pocket_positions = self.pocket_position_fc(global_features)  # [batch_size, 2]

        # 分离起始和结束位置
        pocket_start = pocket_positions[:, 0]  # [batch_size]
        pocket_end = pocket_positions[:, 1]  # [batch_size]

        return affinity, pocket_start, pocket_end




