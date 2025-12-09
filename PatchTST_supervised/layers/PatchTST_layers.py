# 定义模块的公共接口，指定from module import *时导出的内容
__all__ = ['Transpose', 'get_activation_fn', 'moving_avg', 'series_decomp', 
           'PositionalEncoding', 'SinCosPosEncoding', 'Coord2dPosEncoding', 
           'Coord1dPosEncoding', 'positional_encoding']
import torch
from torch import nn
import math

class Transpose(nn.Module):
    """
    维度转置模块：灵活地交换张量的指定维度
    
    参数:
        *dims: 要交换的维度索引，例如 (1, 2) 交换第1和第2维
        contiguous: 是否返回连续内存布局的张量，默认为False
                   设为True可避免某些操作后的内存不连续问题
    """
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        # 执行维度转置，根据contiguous标志决定是否调用contiguous()
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)

def get_activation_fn(activation):
    """
    激活函数工厂：根据输入返回对应的激活函数实例
    参数:activation: 可以是字符串("relu"/"gelu")或可调用对象
    返回:激活函数模块实例(nn.ReLU或nn.GELU)
    实现细节:
        - 支持直接传入可调用对象（如自定义激活函数）
        - 字符串不区分大小写
        - 若传入无效参数则抛出ValueError
    """
    # 如果传入的是可调用的函数/类，直接实例化并返回
    if callable(activation):
        return activation()
    # 字符串匹配，不区分大小写
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    # 不支持的激活函数类型
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')
# ============ 时间序列分解模块 ============
class moving_avg(nn.Module):
    """
    移动平均模块：用于平滑时间序列，突出趋势成分
    
    原理: 通过平均池化(AvgPool)计算滑动窗口内的平均值，抑制高频波动
    
    参数:
        kernel_size: 滑动窗口大小，控制平滑程度
        stride: 步长，通常设为1保持序列长度
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        # 使用一维平均池化实现移动平均，padding=0表示不自动补零
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        前向传播：计算移动平均并处理边界效应
        
        参数:
            x: 输入张量，形状 (B, L, D)，B为batch，L为序列长度，D为特征维度
            
        实现细节:
            1. 对序列两端进行填充，避免边界处窗口不完整
               front/end重复第一个/最后一个值 (kernel_size-1)//2 次
            2. 转置为 (B, D, L) 适应AvgPool1d的输入格式
            3. 应用平均池化
            4. 转置回 (B, L, D)
        """
        # 在序列两端填充，确保边界点也有完整的窗口
        # front: 重复第一个时间步 (kernel_size-1)//2 次
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # end: 重复最后一个时间步 (kernel_size-1)//2 次
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        
        # 拼接填充后的序列: [front, x, end]
        x = torch.cat([front, x, end], dim=1)
        
        # 转置并应用平均池化: (B, L, D) -> (B, D, L) -> 池化 -> (B, D, L')
        x = self.avg(x.permute(0, 2, 1))
        
        # 转置回原始格式: (B, D, L') -> (B, L', D)
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    序列分解模块：将时间序列分解为残差成分和趋势成分
    
    实现: 基于移动平均的思想，趋势 = 平滑后的序列，残差 = 原序列 - 趋势
    
    参数:
        kernel_size: 移动平均的窗口大小
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """
        前向传播：执行序列分解
        
        参数:
            x: 输入张量，形状 (B, L, D)
            
        返回:
            res: 残差成分 (季节性、波动等)，形状 (B, L, D)
            moving_mean: 趋势成分，形状 (B, L, D)
        """
        # 计算趋势（移动平均）
        moving_mean = self.moving_avg(x)
        
        # 残差 = 原序列 - 趋势
        res = x - moving_mean
        
        return res, moving_mean


# ============ 位置编码模块 ============

def PositionalEncoding(q_len, d_model, normalize=True):
    """
    标准Transformer正弦余弦位置编码
    
    数学公式:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    参数:
        q_len: 序列长度
        d_model: 模型维度（特征维度）
        normalize: 是否对编码进行标准化（零均值，标准差缩放）
        
    返回:
        pe: 位置编码矩阵，形状 (q_len, d_model)
    """
    # 初始化全零矩阵
    pe = torch.zeros(q_len, d_model)
    
    # 位置索引: [0, 1, 2, ..., q_len-1]^T
    position = torch.arange(0, q_len).unsqueeze(1)
    
    # 分母项: 10000^(2i/d_model)，使用exp实现
    # torch.arange(0, d_model, 2)生成偶数索引i
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    
    # 偶数维用sin，奇数维用cos
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引
    
    # 标准化：零均值，标准差缩放10倍（使幅值适中）
    if normalize:
        pe = pe - pe.mean()  # 零均值
        pe = pe / (pe.std() * 10)  # 标准差缩放
    
    return pe

# 别名：SinCosPosEncoding是正弦余弦位置编码的另一种名称
SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    """
    2D坐标位置编码：基于坐标的乘积生成位置编码
    
    原理: 编码值 = (2 * (pos_q_len^x * pos_d_model^x) - 1)，通过调整x使均值接近0
    
    参数:
        q_len: 序列长度
        d_model: 模型维度
        exponential: 是否使用指数坐标 (x^0.5) 而非线性坐标 (x^1)
        normalize: 是否标准化
        eps: 均值收敛阈值，当|mean| <= eps时停止调整
        verbose: 是否打印调试信息
        
    实现细节:
        使用迭代调整指数x（从1.0开始），直到编码均值足够接近0
        这是为了平衡不同维度的贡献，避免编码偏向某一维度
    """
    # 初始指数值
    x = .5 if exponential else 1
    i = 0
    
    # 迭代调整指数x，直到编码均值在eps范围内
    for i in range(100):
        # 生成2D网格坐标并计算乘积
        # torch.linspace(0, 1, q_len).reshape(-1, 1): 行向量 (q_len, 1)
        # torch.linspace(0, 1, d_model).reshape(1, -1): 列向量 (1, d_model)
        # 乘积得到 (q_len, d_model) 的坐标矩阵
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * \
              (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        
        # 打印调试信息（如果verbose=True）
        if verbose:
            print(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}')
        
        # 检查均值是否收敛
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += .001  # 均值过大，增大x
        else:
            x -= .001  # 均值过小，减小x
        i += 1
    
    # 标准化处理
    if normalize:
        cpe = cpe - cpe.mean()  # 零均值
        cpe = cpe / (cpe.std() * 10)  # 标准差缩放
    
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    """
    1D坐标位置编码：仅基于序列位置生成编码
    
    参数:
        q_len: 序列长度
        exponential: 是否使用指数坐标
        normalize: 是否标准化
        
    返回:
        cpe: 1D位置编码，形状 (q_len, 1)
    """
    # 生成1D坐标并转换到[-1, 1]范围
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (.5 if exponential else 1)) - 1)
    
    # 标准化
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    """
    位置编码工厂函数：根据配置生成不同类型的位置编码
    
    参数:
        pe: 位置编码类型字符串或None
        learn_pe: 是否将位置编码作为可学习参数
        q_len: 序列长度
        d_model: 模型维度
        
    返回:
        nn.Parameter: 位置编码参数，形状 (q_len, d_model) 或 (q_len, 1)
        
    支持的pe类型:
        - None: 随机初始化，uniform[-0.02, 0.02]
        - 'zero': 单通道zeros初始化
        - 'zeros': 全通道zeros初始化
        - 'normal'/'gauss': 高斯初始化
        - 'uniform': 均匀初始化
        - 'lin1d': 线性1D坐标编码
        - 'exp1d': 指数1D坐标编码
        - 'lin2d': 线性2D坐标编码
        - 'exp2d': 指数2D坐标编码
        - 'sincos': 正弦余弦编码
    """
    # 根据pe类型选择不同的初始化方式
    if pe == None:
        # None类型：随机初始化，用于测试位置编码的影响（learn_pe=False时不更新）
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False  # 强制设为不可学习
    
    elif pe == 'zero':
        # zero：单通道zeros，通常用于消融实验
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    
    elif pe == 'zeros':
        # zeros：全通道zeros，uniform初始化
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    
    elif pe == 'normal' or pe == 'gauss':
        # 高斯分布初始化
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    
    elif pe == 'uniform':
        # 均匀分布初始化
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    
    # 坐标编码类型
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    
    # 正弦余弦编码
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    
    else:
        # 不支持的类型，抛出错误
        raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
            'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    
    # 返回nn.Parameter，根据learn_pe控制是否需要梯度
    return nn.Parameter(W_pos, requires_grad=learn_pe)
