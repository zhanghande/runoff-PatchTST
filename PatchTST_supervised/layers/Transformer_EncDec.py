import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """卷积下采样层：对序列进行局部特征提取和降维"""
    
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        # 1D卷积：通道不变，kernel=3，circular padding处理周期性特征
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        # 批归一化稳定训练
        self.norm = nn.BatchNorm1d(c_in)
        # ELU激活函数
        self.activation = nn.ELU()
        # 最大池化：stride=2实现序列长度减半
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: (B, L, D) -> 转置为 (B, D, L) 以适应Conv1d
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)  # 序列长度减半
        # 转置回 (B, L', D)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """编码器层：自注意力 + 前馈网络（卷积实现）"""
    
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # 默认FFN维度为4倍模型维度
        self.attention = attention  # 自注意力模块
        # 两个1x1卷积实现前馈网络（替代全连接，效率更高）
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)  # 注意力后归一化
        self.norm2 = nn.LayerNorm(d_model)  # FFN后归一化
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # 自注意力 + 残差连接
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        # 前馈网络（卷积实现）
        y = x = self.norm1(x)  # 预归一化
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # (B, D, L) -> (B, D_ff, L)
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # (B, D_ff, L) -> (B, L, D)

        return self.norm2(x + y), attn  # 残差连接 + 归一化


class Encoder(nn.Module):
    """编码器：堆叠多个编码器层，可选插入卷积下采样层"""
    
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)  # 注意力层列表
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None  # 卷积层列表
        self.norm = norm_layer  # 最终归一化层

    def forward(self, x, attn_mask=None):
        # x: [B, L, D]
        attns = []  # 存储注意力权重
        if self.conv_layers is not None:
            # 交替执行注意力和卷积（每层后降采样）
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)  # 序列长度减半
                attns.append(attn)
            # 处理最后一个注意力层（无后续卷积）
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            # 纯注意力堆叠（无卷积）
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        # 最终层归一化
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """解码器层：自注意力 + 交叉注意力 + 前馈网络"""
    
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention  # 自注意力（带掩码）
        self.cross_attention = cross_attention  # 交叉注意力（编码器-解码器）
        # 前馈网络（同EncoderLayer）
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力后
        self.norm2 = nn.LayerNorm(d_model)  # 交叉注意力后
        self.norm3 = nn.LayerNorm(d_model)  # FFN后
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # 自注意力（带目标序列掩码）+ 残差连接
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        # 交叉注意力（编码器输出作为K,V）+ 残差连接
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        # 前馈网络
        y = x = self.norm2(x)  # 预归一化
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)  # 残差连接 + 归一化


class Decoder(nn.Module):
    """解码器：堆叠多个解码器层，输出投影"""
    
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)  # 解码器层列表
        self.norm = norm_layer  # 最终归一化
        self.projection = projection  # 输出投影层（如词表映射）

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # 逐层前向传播
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        
        # 最终归一化
        if self.norm is not None:
            x = self.norm(x)
            
        # 投影到输出空间（如词汇表大小）
        if self.projection is not None:
            x = self.projection(x)
        return x
