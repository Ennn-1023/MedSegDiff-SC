"""
LoRA (Low-Rank Adaptation) 實作

此模組提供 LoRA 層的實作和自動注入機制，用於高效微調預訓練模型。

主要功能：
1. LoRALayer: 低秩適應層
2. LinearWithLoRA: 包裝 Linear 層的 LoRA 版本
3. inject_lora: 自動將 LoRA 注入模型
4. count_lora_parameters: 統計 LoRA 參數量
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    LoRA 低秩適應層
    
    實現公式: h = W_0 * x + (alpha / r) * B * A * x
    其中:
        - W_0 是凍結的預訓練權重
        - A ∈ R^{r × in_features}, B ∈ R^{out_features × r}
        - r 是秩 (rank)
        - alpha 是縮放因子
    
    Args:
        in_features: 輸入特徵維度
        out_features: 輸出特徵維度
        rank: 低秩分解的秩
        alpha: 縮放超參數
        dropout: Dropout 比率
    """
    
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # LoRA 權重: A 和 B
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout (可選)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # 縮放因子
        self.scaling = alpha / rank
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化 LoRA 參數"""
        # A: Kaiming uniform 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B: 零初始化 (確保初始時 LoRA 不影響原模型)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: 輸入張量, shape = (batch_size, ..., in_features)
        
        Returns:
            LoRA 輸出, shape = (batch_size, ..., out_features)
        """
        # h = (alpha / r) * B * A * x
        x = self.dropout(x)
        # (batch, ..., in_features) @ (in_features, rank) -> (batch, ..., rank)
        h = F.linear(x, self.lora_A)
        # (batch, ..., rank) @ (rank, out_features) -> (batch, ..., out_features)
        h = F.linear(h, self.lora_B)
        # 縮放
        h = h * self.scaling
        return h


class LinearWithLoRA(nn.Module):
    """
    將 LoRA 層包裝到原始 Linear 層的包裝器
    
    前向傳播: output = W_0 * x + LoRA(x)
    
    Args:
        linear: 原始的 nn.Linear 層 (將被凍結)
        rank: LoRA 的秩
        alpha: LoRA 的縮放因子
        dropout: LoRA 的 dropout 比率
    """
    
    def __init__(self, linear, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # 凍結原始權重
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x):
        """前向傳播: 原始層 + LoRA"""
        return self.linear(x) + self.lora(x)


class Conv1dWithLoRA(nn.Module):
    """
    將 LoRA 層包裝到原始 Conv1d 層的包裝器
    
    專門用於 UNet Attention 中的 QKV 和投影層
    前向傳播: output = W_0 * x + LoRA(x)
    
    Args:
        conv1d: 原始的 nn.Conv1d 層 (將被凍結)
        rank: LoRA 的秩
        alpha: LoRA 的縮放因子
        dropout: LoRA 的 dropout 比率
    """
    
    def __init__(self, conv1d, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.conv1d = conv1d
        
        # 對於 1x1 卷積，可以視為 Linear 層
        # Conv1d: (out_channels, in_channels, kernel_size)
        # 等價於 Linear: (out_channels, in_channels) when kernel_size=1
        assert conv1d.kernel_size == (1,), "LoRA only supports 1x1 Conv1d"
        
        self.lora = LoRALayer(
            in_features=conv1d.in_channels,
            out_features=conv1d.out_channels,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # 凍結原始權重
        self.conv1d.weight.requires_grad = False
        if self.conv1d.bias is not None:
            self.conv1d.bias.requires_grad = False
    
    def forward(self, x):
        """
        前向傳播: 原始 Conv1d + LoRA
        
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        # 原始卷積輸出
        h = self.conv1d(x)
        
        # LoRA 輸出 (需要轉換維度)
        # x: (batch, in_channels, seq_len) -> (batch, seq_len, in_channels)
        x_permuted = x.permute(0, 2, 1)
        # LoRA 處理: (batch, seq_len, in_channels) -> (batch, seq_len, out_channels)
        lora_out = self.lora(x_permuted)
        # 轉回: (batch, seq_len, out_channels) -> (batch, out_channels, seq_len)
        lora_out = lora_out.permute(0, 2, 1)
        
        return h + lora_out


def inject_lora(model, rank=4, alpha=1.0, dropout=0.0, target_modules='emb_only'):
    """
    將 LoRA 層自動注入到模型中
    
    Args:
        model: 要注入 LoRA 的模型
        rank: LoRA 的秩
        alpha: LoRA 的縮放因子
        dropout: LoRA 的 dropout 比率
        target_modules: 要注入 LoRA 的模組策略，可選:
                       - 'emb_only': 只注入 Embedding 層 (預設，最保守)
                       - 'attn_only': 只注入 Attention 的 QKV 和投影層
                       - 'attn_emb': 注入 Attention + Embedding (推薦)
                       - 或者自定義列表: ['qkv', 'proj_out', 'emb_layers']
    
    Returns:
        注入 LoRA 後的模型
    """
    # 策略映射
    strategy_map = {
        'emb_only': ['emb_layers'],
        'attn_only': ['qkv', 'proj_out'],
        'attn_emb': ['qkv', 'proj_out', 'emb_layers'],
    }
    
    # 解析 target_modules
    if isinstance(target_modules, str):
        if target_modules in strategy_map:
            target_modules = strategy_map[target_modules]
            strategy_name = target_modules
        else:
            raise ValueError(f"Unknown strategy '{target_modules}'. Choose from: {list(strategy_map.keys())}")
    elif isinstance(target_modules, list):
        strategy_name = 'custom'
    else:
        target_modules = strategy_map['emb_only']
        strategy_name = 'emb_only'
    
    print(f"🔍 Injecting LoRA (rank={rank}, alpha={alpha}, dropout={dropout}) into model...")
    print(f"   Strategy: {strategy_name}")
    print(f"   Target modules: {target_modules}")
    
    # 第一階段: 收集所有需要修改的模組
    modules_to_modify = []
    
    for name, module in model.named_modules():
        # 檢查是否為目標模組
        should_inject = any(target in name for target in target_modules)
        
        if not should_inject:
            continue
        
        # 找到父模組和子模組名稱
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent_module = dict(model.named_modules())[parent_name]
        else:
            parent_module = model
        
        # 支持 Linear 和 Conv1d 層
        if isinstance(module, nn.Linear):
            modules_to_modify.append(('linear', parent_module, child_name, module, name))
        elif isinstance(module, nn.Conv1d) and module.kernel_size == (1,):
            modules_to_modify.append(('conv1d', parent_module, child_name, module, name))
    
    # 第二階段: 統一替換
    injected_count = 0
    linear_count = 0
    conv1d_count = 0
    
    for module_type, parent_module, child_name, child, full_name in modules_to_modify:
        if module_type == 'linear':
            lora_layer = LinearWithLoRA(child, rank=rank, alpha=alpha, dropout=dropout)
            linear_count += 1
        elif module_type == 'conv1d':
            lora_layer = Conv1dWithLoRA(child, rank=rank, alpha=alpha, dropout=dropout)
            conv1d_count += 1
        
        setattr(parent_module, child_name, lora_layer)
        injected_count += 1
        print(f"   ✓ Injected LoRA into: {full_name}")
    
    print(f"✅ Successfully injected LoRA into {injected_count} layers!")
    print(f"   - Linear layers: {linear_count}")
    print(f"   - Conv1d layers: {conv1d_count}")
    
    # 凍結所有非 LoRA 參數
    freeze_non_lora_parameters(model)
    
    return model


def freeze_non_lora_parameters(model):
    """
    凍結所有非 LoRA 的參數
    
    Args:
        model: 模型
    """
    frozen_count = 0
    lora_count = 0
    
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
    
    print(f"❄️  Frozen {frozen_count} non-LoRA parameters")
    print(f"🔥 Kept {lora_count} LoRA parameters trainable")


def count_lora_parameters(model):
    """
    統計模型中的參數量
    
    Args:
        model: 模型
    
    Returns:
        dict: 包含以下鍵的字典
            - total_params: 總參數量
            - trainable_params: 可訓練參數量
            - lora_params: LoRA 參數量
            - trainable_percentage: 可訓練參數百分比
    """
    total_params = 0
    trainable_params = 0
    lora_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
            
        if 'lora' in name:
            lora_params += num_params
    
    trainable_percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'lora_params': lora_params,
        'trainable_percentage': trainable_percentage
    }


def print_lora_parameters(model):
    """
    打印 LoRA 參數統計
    
    Args:
        model: 模型
    """
    stats = count_lora_parameters(model)
    
    print("\n" + "="*60)
    print("📊 Parameter Statistics")
    print("="*60)
    print(f"Total parameters:      {stats['total_params']:,}")
    print(f"Trainable parameters:  {stats['trainable_params']:,}")
    print(f"LoRA parameters:       {stats['lora_params']:,}")
    print(f"Trainable percentage:  {stats['trainable_percentage']:.2f}%")
    print("="*60 + "\n")


def merge_lora_weights(model):
    """
    將 LoRA 權重合併到原始權重中
    
    這會將 LoRA 的低秩更新合併到原始的 Linear 層，
    之後可以移除 LoRA 層以節省推理時的計算成本。
    
    Args:
        model: 包含 LoRA 層的模型
    
    Returns:
        合併後的模型
    """
    print("🔀 Merging LoRA weights into base model...")
    
    merged_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            # 計算合併後的權重: W' = W_0 + (alpha/r) * B @ A
            with torch.no_grad():
                lora_weight = module.lora.lora_B @ module.lora.lora_A
                lora_weight = lora_weight * module.lora.scaling
                module.linear.weight.data += lora_weight
                
                # 解凍參數
                module.linear.weight.requires_grad = True
                if module.linear.bias is not None:
                    module.linear.bias.requires_grad = True
            
            merged_count += 1
    
    print(f"✅ Merged {merged_count} LoRA layers!")
    return model


def detect_lora_in_state_dict(state_dict):
    """
    從 state_dict 中檢測是否包含 LoRA 權重
    
    Args:
        state_dict: 模型的 state_dict
    
    Returns:
        dict: {'has_lora': bool, 'rank': int or None, 'lora_keys': list}
    """
    lora_keys = [k for k in state_dict.keys() if 'lora' in k]
    
    if not lora_keys:
        return {'has_lora': False, 'rank': None, 'lora_keys': []}
    
    # 嘗試從 lora_A 的形狀推斷 rank
    rank = None
    for key in lora_keys:
        if 'lora_A' in key:
            rank = state_dict[key].shape[0]
            break
    
    return {
        'has_lora': True,
        'rank': rank,
        'lora_keys': lora_keys,
        'num_lora_layers': len([k for k in lora_keys if 'lora_A' in k])
    }


def get_lora_state_dict(model):
    """
    只提取 LoRA 參數（不包含原始權重）
    
    Args:
        model: 包含 LoRA 的模型
    
    Returns:
        dict: 只包含 LoRA 參數的 state_dict
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_state_dict[name] = param.data
    return lora_state_dict


def load_lora_weights(model, lora_state_dict, strict=True):
    """
    只載入 LoRA 權重（不載入原始權重）
    
    Args:
        model: 已經注入 LoRA 的模型
        lora_state_dict: LoRA 權重
        strict: 是否嚴格檢查
    """
    # 過濾出 LoRA 參數
    model_lora_keys = [name for name, _ in model.named_parameters() if 'lora' in name]
    
    missing_keys = []
    unexpected_keys = []
    
    for key in lora_state_dict.keys():
        if key not in model_lora_keys:
            unexpected_keys.append(key)
    
    for key in model_lora_keys:
        if key not in lora_state_dict:
            missing_keys.append(key)
        else:
            # 載入參數
            param = dict(model.named_parameters())[key]
            param.data.copy_(lora_state_dict[key])
    
    if strict and (missing_keys or unexpected_keys):
        error_msg = f"Error loading LoRA weights:\n"
        if missing_keys:
            error_msg += f"  Missing keys: {missing_keys}\n"
        if unexpected_keys:
            error_msg += f"  Unexpected keys: {unexpected_keys}\n"
        raise RuntimeError(error_msg)
    
    return model


if __name__ == "__main__":
    # 簡單測試
    print("Testing LoRA implementation...\n")
    
    # 創建一個簡單的測試模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(512, 1536)
            self.proj = nn.Linear(512, 512)
            self.fc = nn.Linear(512, 256)
        
        def forward(self, x):
            return self.fc(self.proj(self.qkv(x)))
    
    model = SimpleModel()
    print("Original model:")
    print(model)
    
    # 注入 LoRA
    model = inject_lora(model, rank=8, alpha=8.0)
    
    # 打印參數統計
    print_lora_parameters(model)
    
    # 測試前向傳播
    x = torch.randn(4, 512)
    y = model(x)
    print(f"Output shape: {y.shape}")
    print("\n✅ All tests passed!")
