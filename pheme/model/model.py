import os
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import math
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def check_disk_space(path='.', min_gb=2):
    """检查磁盘空间"""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        if free_gb < min_gb:
            logger.warning(f"⚠️  磁盘空间不足: {free_gb:.2f}GB (需要至少 {min_gb}GB)")
            return False
        logger.info(f"✓ 磁盘空间充足: {free_gb:.2f}GB 可用")
        return True
    except Exception as e:
        logger.warning(f"无法检查磁盘空间: {e}")
        return True

def save_checkpoint_safely(checkpoint, save_path, max_retries=3):
    """安全保存模型(带重试机制)"""
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            temp_path = save_path + '.tmp'
            torch.save(checkpoint, temp_path)
            
            if os.path.exists(save_path):
                backup_path = save_path + '.backup'
                shutil.move(save_path, backup_path)
            
            shutil.move(temp_path, save_path)
            
            backup_path = save_path + '.backup'
            if os.path.exists(backup_path):
                os.remove(backup_path)
            
            logger.info(f"✓ 模型已安全保存到 {save_path}")
            return True
            
        except Exception as e:
            logger.warning(f"保存失败 (尝试 {attempt+1}/{max_retries}): {e}")
            
            temp_path = save_path + '.tmp'
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logger.error(f"✗ 保存模型失败: {e}")
                try:
                    json_path = save_path.replace('.pth', '_backup.json')
                    with open(json_path, 'w') as f:
                        json.dump({
                            'val_metric': float(checkpoint.get('val_metric', 0)),
                            'epoch': checkpoint.get('epoch', 0),
                            'error': str(e)
                        }, f, indent=2)
                    logger.info(f"✓ 关键信息已保存到 {json_path}")
                except Exception as json_err:
                    logger.error(f"保存备份JSON也失败: {json_err}")
                
                return False
    
    return False

# ==================== 梯度反转层 ====================
class GradientReversalFunction(Function):
    """梯度反转层 - 实现对抗训练的核心"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

# ==================== 模态解耦模块 ====================
class ModalityDisentanglement(nn.Module):
    """模态解耦模块 - 使用对抗训练分离共享和特定特征"""
    def __init__(self, feature_dim, shared_dim):
        super().__init__()
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(feature_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(shared_dim, shared_dim)
        )
        
        self.specific_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim - shared_dim),
            nn.LayerNorm(feature_dim - shared_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim - shared_dim, feature_dim - shared_dim)
        )
        
        self.modality_discriminator = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.gradient_reversal = GradientReversalLayer(alpha=1.0)
        
    def forward(self, text_features, image_features, return_discriminator_loss=True):
        text_shared = self.shared_encoder(text_features)
        image_shared = self.shared_encoder(image_features)
        
        text_specific = self.specific_encoder(text_features)
        image_specific = self.specific_encoder(image_features)
        
        if return_discriminator_loss:
            text_pred = self.modality_discriminator(text_shared.detach())
            image_pred = self.modality_discriminator(image_shared.detach())
            
            discriminator_loss = (
                F.binary_cross_entropy_with_logits(text_pred, torch.zeros_like(text_pred)) +
                F.binary_cross_entropy_with_logits(image_pred, torch.ones_like(image_pred))
            ) / 2
        else:
            discriminator_loss = torch.tensor(0.0).to(text_features.device)
        
        reversed_text = self.gradient_reversal(text_shared)
        reversed_image = self.gradient_reversal(image_shared)
        
        adv_text_pred = self.modality_discriminator(reversed_text)
        adv_image_pred = self.modality_discriminator(reversed_image)
        
        adversarial_loss = (
            F.binary_cross_entropy_with_logits(adv_text_pred, torch.ones_like(adv_text_pred) * 0.5) +
            F.binary_cross_entropy_with_logits(adv_image_pred, torch.ones_like(adv_image_pred) * 0.5)
        ) / 2
        
        text_combined = torch.cat([text_shared, text_specific], dim=-1)
        image_combined = torch.cat([image_shared, image_specific], dim=-1)
        
        return text_combined, image_combined, discriminator_loss, adversarial_loss

# ==================== 对比学习模块 ====================
class ContrastiveLearningModule(nn.Module):
    """简化的对比学习模块"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def info_nce_loss(self, text_features, image_features):
        text_features = F.normalize(text_features, dim=1)
        image_features = F.normalize(image_features, dim=1)
        
        batch_size = text_features.size(0)
        logits = torch.matmul(text_features, image_features.t()) / self.temperature
        labels = torch.arange(batch_size).to(text_features.device)
        
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.t(), labels)
        
        loss = (loss_t2i + loss_i2t) / 2
        return loss

# ==================== 引导注意力模块 ====================
class GuidedAttentionModule(nn.Module):
    """引导注意力模块"""
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, 8, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, text_features, image_features):
        attended_text, _ = self.attention(text_features, image_features, image_features)
        gate = self.gate(torch.cat([text_features, attended_text], dim=-1))
        enhanced_features = gate * attended_text + (1 - gate) * text_features
        return enhanced_features

# ==================== 多模态融合模块 ====================
class MultiModalFusion(nn.Module):
    """多模态融合模块"""
    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int, shared_dim: int):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        
        self.text_image_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.image_text_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        
        self.disentanglement = ModalityDisentanglement(hidden_dim, shared_dim)
        self.guided_attention = GuidedAttentionModule(hidden_dim)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> Tuple:
        text_proj = self.text_projection(text_embeddings).unsqueeze(1)
        image_proj = self.image_projection(image_embeddings).unsqueeze(1)
        
        text_attended, _ = self.text_image_attention(text_proj, image_proj, image_proj)
        image_attended, _ = self.image_text_attention(image_proj, text_proj, text_proj)
        
        text_cls = text_attended.squeeze(1)
        image_cls = image_attended.squeeze(1)
        
        text_combined, image_combined, disc_loss, adv_loss = self.disentanglement(
            text_cls, image_cls, return_discriminator_loss=True
        )
        
        enhanced_text = self.guided_attention(
            text_combined.unsqueeze(1), 
            image_combined.unsqueeze(1)
        ).squeeze(1)
        
        fused_features = self.fusion_layer(torch.cat([enhanced_text, image_combined], dim=1))
        
        return fused_features, disc_loss, adv_loss

# ==================== 跨模态融合模块 ====================
class CrossModalFusionModule(nn.Module):
    """跨模态融合模块"""
    def __init__(self, text_image_dim, propagation_dim, fusion_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.text_image_projection = nn.Linear(text_image_dim, fusion_dim)
        self.propagation_projection = nn.Linear(propagation_dim, fusion_dim)
        
        self.cross_attention1 = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attention2 = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True)
            
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.norm3 = nn.LayerNorm(fusion_dim)
        self.norm4 = nn.LayerNorm(fusion_dim)
        
        self.ffn1 = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim)
        )
        
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, text_image_repr, propagation_repr):
        text_image = self.text_image_projection(text_image_repr)
        propagation = self.propagation_projection(propagation_repr)
        
        text_image = text_image.unsqueeze(1)
        propagation = propagation.unsqueeze(1)
        
        attn_output1, _ = self.cross_attention1(text_image, propagation, propagation)
        text_image = self.norm1(text_image + attn_output1)
        text_image = self.norm2(text_image + self.ffn1(text_image))
        
        attn_output2, _ = self.cross_attention2(propagation, text_image, text_image)
        propagation = self.norm3(propagation + attn_output2)
        propagation = self.norm4(propagation + self.ffn2(propagation))
        
        text_image = text_image.squeeze(1)
        propagation = propagation.squeeze(1)
        
        combined = torch.cat([text_image, propagation], dim=1)
        fused = self.final_fusion(combined)
        
        return fused

# ==================== 传播链处理模块 ====================
class LayerDecayAttention(nn.Module):
    """层级衰减注意力"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, depth_masks, chain_mask=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        q = self.q_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        depth_masks = depth_masks.unsqueeze(1).unsqueeze(2)
        attention_scores = attention_scores * depth_masks
        
        if chain_mask is not None:
            chain_mask = chain_mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(chain_mask == 0, float('-inf'))
        
        attn = torch.softmax(attention_scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_length, self.hidden_size)
        
        return self.out(out)

class TreeTransformerLayer(nn.Module):
    """树形Transformer层"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = LayerDecayAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, depth_masks, chain_mask=None):
        attn_output = self.attention(x, depth_masks, chain_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# ==================== 主模型 ====================
class IntegratedRumorDetectionModel(nn.Module):
    """集成谣言检测模型"""
    def __init__(self, text_embedding_dim, image_embedding_dim, 
                 propagation_embedding_dim, num_classes=2):
        super().__init__()
        
        self.text_embedding_dim = text_embedding_dim
        self.image_embedding_dim = image_embedding_dim
        self.propagation_embedding_dim = propagation_embedding_dim
        
        hidden_dim = 512
        shared_dim = 256
        
        # 文本-图像融合模块
        self.text_image_fusion = MultiModalFusion(
            text_embedding_dim, image_embedding_dim, hidden_dim, shared_dim
        )
        self.contrastive = ContrastiveLearningModule(temperature=0.07)
        
        # 传播链处理模块
        self.propagation_adapter = nn.Sequential(
            nn.Linear(propagation_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.transformer_layers = nn.ModuleList([
            TreeTransformerLayer(hidden_dim, num_heads=8, dropout=0.1)
            for _ in range(3)
        ])
        
        # 交叉模态融合
        self.cross_fusion = CrossModalFusionModule(
            text_image_dim=hidden_dim,
            propagation_dim=hidden_dim,
            fusion_dim=hidden_dim
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, text_embeddings, image_embeddings,
                chain_embeddings=None, depth_info=None, chain_structure=None):
        contrastive_loss = torch.tensor(0.0).to(text_embeddings.device)
        disc_loss = torch.tensor(0.0).to(text_embeddings.device)
        adv_loss = torch.tensor(0.0).to(text_embeddings.device)
        
        # 对比学习损失
        text_proj = self.text_image_fusion.text_projection(text_embeddings)
        image_proj = self.text_image_fusion.image_projection(image_embeddings)
        contrastive_loss = self.contrastive.info_nce_loss(text_proj, image_proj)
        
        # 完整融合
        text_image_features, disc_loss_ti, adv_loss_ti = self.text_image_fusion(
            text_embeddings, image_embeddings
        )
        
        disc_loss += disc_loss_ti
        adv_loss += adv_loss_ti
        
        # 传播链表征
        if chain_embeddings is not None:
            chain_embeddings = self.propagation_adapter(chain_embeddings)
            batch_size, max_length, _ = chain_embeddings.size()
            
            depth_masks = self._create_depth_masks(depth_info, max_length, chain_embeddings.device)
            chain_mask = self._create_chain_mask(
                chain_structure, max_length, batch_size, chain_embeddings.device
            )
            
            x = chain_embeddings
            for layer in self.transformer_layers:
                x = layer(x, depth_masks, chain_mask)
            
            propagation_features = x[:, 0, :]
            
            # 交叉模态融合
            fused_features = self.cross_fusion(text_image_features, propagation_features)
        else:
            fused_features = text_image_features
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits, contrastive_loss, disc_loss, adv_loss
    
    def _create_depth_masks(self, depth_info, max_length, device):
        """创建深度衰减掩码"""
        batch_size = len(depth_info)
        depth_masks = torch.zeros((batch_size, max_length)).to(device)
        
        for i, depths in enumerate(depth_info):
            for j, depth in enumerate(depths):
                if j < max_length:
                    depth_masks[i][j] = torch.exp(-torch.tensor(depth, dtype=torch.float32) * 0.05)
        
        return depth_masks
    
    def _create_chain_mask(self, chain_structure, max_length, batch_size, device):
        """创建传播链结构掩码"""
        chain_mask = torch.zeros((batch_size, max_length, max_length)).to(device)
        
        for i, structure in enumerate(chain_structure):
            if not structure or len(structure) == 0:
                chain_mask[i, 0, 0] = 1
                continue
            
            valid_nodes = set(structure.keys())
            for child_list in structure.values():
                valid_nodes.update(child_list)
            
            valid_nodes = {n for n in valid_nodes if isinstance(n, int) and n < max_length}
            
            if not valid_nodes:
                chain_mask[i, 0, 0] = 1
                continue
            
            for node_idx in valid_nodes:
                if node_idx < max_length:
                    chain_mask[i][node_idx][node_idx] = 1
            
            for parent_idx, children_indices in structure.items():
                if not isinstance(parent_idx, int) or parent_idx >= max_length:
                    continue
                
                for child_idx in children_indices:
                    if isinstance(child_idx, int) and child_idx < max_length:
                        chain_mask[i][parent_idx][child_idx] = 1
                        chain_mask[i][child_idx][parent_idx] = 1
        
        return chain_mask

# ==================== 数据集 ====================
class RumorDataset(Dataset):
    """谣言检测数据集"""
    def __init__(self, data_dict: Dict):
        self.ids = list(data_dict.keys())
        self.data = data_dict
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        mid = self.ids[idx]
        item = self.data[mid]
        
        return {
            'text_embedding': torch.tensor(item['text_embedding'], dtype=torch.float32),
            'image_embedding': torch.tensor(item['image_embedding'], dtype=torch.float32),
            'chains': item['chain'],
            'depths': item['depth'],
            'structures': item['structure'],
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

def collate_fn(batch: List[Dict]):
    """批处理函数"""
    text_embeddings = torch.stack([item['text_embedding'] for item in batch])
    image_embeddings = torch.stack([item['image_embedding'] for item in batch])
    chains = [item['chains'] for item in batch]
    depths = [item['depths'] for item in batch]
    structures = [item['structures'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # Pad chains
    if len(chains) > 0 and chains[0].size(0) > 0:
        max_len = max(chain.size(0) for chain in chains)
        max_len = min(max_len, 100)
        
        padded_chains = []
        for chain in chains:
            if chain.size(0) < max_len:
                padding = torch.zeros(max_len - chain.size(0), chain.size(1))
                padded_chain = torch.cat([chain, padding], dim=0)
                padded_chains.append(padded_chain)
            else:
                padded_chains.append(chain[:max_len])
        
        padded_chains = torch.stack(padded_chains)
    else:
        padded_chains = torch.zeros(len(batch), 1, chains[0].size(1) if len(chains) > 0 else 768)
    
    return {
        'text_embeddings': text_embeddings,
        'image_embeddings': image_embeddings,
        'chains': padded_chains,
        'depths': depths,
        'structures': structures,
        'label': labels
    }

# ==================== 训练器 ====================
class RumorTrainer:
    """训练器（标准CE Loss）"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, args):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        
        # Loss权重
        self.contrastive_weight = 0.1
        self.disc_weight = 0.01
        self.adv_weight = 0.01
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器 - ReduceLROnPlateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        # 标准交叉熵损失
        self.criterion = nn.CrossEntropyLoss()
        logger.info("使用标准 CrossEntropyLoss")
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        all_labels = []
        all_predictions = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            text_embs = batch['text_embeddings'].to(self.device)
            image_embs = batch['image_embeddings'].to(self.device)
            chains = batch['chains'].to(self.device)
            depths = batch['depths']
            structures = batch['structures']
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, contrastive_loss, disc_loss, adv_loss = self.model(
                text_embs, image_embs, chains, depths, structures
            )
            
            cls_loss = self.criterion(logits, labels)
            loss = (
                cls_loss + 
                self.contrastive_weight * contrastive_loss +
                self.disc_weight * disc_loss +
                self.adv_weight * adv_loss
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss at batch {batch_idx}, skipping...")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = logits.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        
        num_batches = len(self.train_loader)
        metrics = calculate_metrics(
            np.array(all_labels), 
            np.array(all_predictions), 
            phase="train",
            verbose=False
        )
        
        return total_loss / num_batches, metrics
    
    def evaluate(self, data_loader, phase="val", verbose=True):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                text_embs = batch['text_embeddings'].to(self.device)
                image_embs = batch['image_embeddings'].to(self.device)
                chains = batch['chains'].to(self.device)
                depths = batch['depths']
                structures = batch['structures']
                labels = batch['label'].to(self.device)
                
                logits, contrastive_loss, disc_loss, adv_loss = self.model(
                    text_embs, image_embs, chains, depths, structures
                )
                
                cls_loss = self.criterion(logits, labels)
                batch_loss = (
                    cls_loss + 
                    self.contrastive_weight * contrastive_loss +
                    self.disc_weight * disc_loss +
                    self.adv_weight * adv_loss
                )
                
                total_loss += batch_loss.item()
                
                _, predicted = logits.max(1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        metrics = calculate_metrics(
            np.array(all_labels), 
            np.array(all_predictions), 
            phase=phase,
            verbose=verbose
        )
        return total_loss / len(data_loader), metrics

# ==================== 评估指标 ====================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     phase: str = "eval", verbose: bool = True) -> Dict[str, float]:
    """计算评估指标"""
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty predictions or labels!")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro平均指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 详细打印
    if verbose:
        true_counts = dict(zip(*np.unique(y_true, return_counts=True)))
        pred_counts = dict(zip(*np.unique(y_pred, return_counts=True)))
        
        logger.info(f"\n{'='*70}")
        logger.info(f"{phase.upper()} - 详细评估指标")
        logger.info(f"{'='*70}")
        logger.info(f"真实标签分布: Class 0={true_counts.get(0, 0)}, Class 1={true_counts.get(1, 0)}")
        logger.info(f"预测标签分布: Class 0={pred_counts.get(0, 0)}, Class 1={pred_counts.get(1, 0)}")
        
        logger.info(f"\n混淆矩阵:")
        logger.info(f"{'':14} Pred 0    Pred 1")
        logger.info(f"Actual 0 (真) {cm[0,0]:7d}    {cm[0,1]:7d}")
        logger.info(f"Actual 1 (谣) {cm[1,0]:7d}    {cm[1,1]:7d}")
        
        logger.info(f"\n汇总指标:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        logger.info(f"{'='*70}\n")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

# ==================== 数据处理函数 ====================
def load_precomputed_embeddings(embeddings_file, image_embeddings_file):
    """加载预计算的embeddings"""
    logger.info("="*70)
    logger.info("加载预计算的embeddings...")
    
    with open(embeddings_file, 'r') as f:
        text_embeddings = json.load(f)
    
    with open(image_embeddings_file, 'r') as f:
        image_embeddings = json.load(f)
    
    text_emb_dict = {k: np.array(v, dtype=np.float32) for k, v in text_embeddings.items()}
    image_emb_dict = {k: np.array(v, dtype=np.float32) for k, v in image_embeddings.items()}
    
    logger.info(f"✓ 加载了 {len(text_emb_dict)} 个文本embeddings")
    logger.info(f"✓ 加载了 {len(image_emb_dict)} 个图像embeddings")
    
    text_dim = len(next(iter(text_emb_dict.values())))
    image_dim = len(next(iter(image_emb_dict.values())))
    
    logger.info(f"✓ 文本embedding维度: {text_dim}")
    logger.info(f"✓ 图像embedding维度: {image_dim}")
    logger.info("="*70)
    
    return text_emb_dict, image_emb_dict, text_dim, image_dim

def build_propagation_tree(post_id, reply_dict, embeddings_dict, visited=None, max_depth=10, current_depth=0):
    """构建传播树"""
    if visited is None:
        visited = set()
    
    if current_depth >= max_depth:
        return {}, [], {}
    
    if post_id in visited:
        return {}, [], {}
    
    visited.add(post_id)
    post_id = str(post_id)
    
    try:
        if post_id in embeddings_dict['original_posts']:
            current_embedding = embeddings_dict['original_posts'][post_id]
        elif post_id in embeddings_dict['reply_posts']:
            current_embedding = embeddings_dict['reply_posts'][post_id]
        else:
            return {}, [], {}
            
        replies = reply_dict.get(post_id, [])
        
        tree = {post_id: []}
        embeddings = [current_embedding]
        depths = {post_id: 0}
        
        for reply_id in replies:
            sub_tree, sub_embeddings, sub_depths = build_propagation_tree(
                reply_id, reply_dict, embeddings_dict, visited, max_depth, current_depth + 1
            )
            if sub_tree:
                tree[post_id].append(reply_id)
                tree.update(sub_tree)
                embeddings.extend(sub_embeddings)
                
                for node, depth in sub_depths.items():
                    depths[node] = depth + 1
        
        return tree, embeddings, depths
        
    except Exception as e:
        logger.error(f"Error processing post_id {post_id}: {str(e)}")
        return {}, [], {}

def load_and_process_data(embeddings_file, post_csv, comment_csv):
    """加载并处理传播链数据"""
    logger.info("="*70)
    logger.info("加载传播链数据...")
    
    with open(embeddings_file, 'r') as f:
        embeddings_dict = json.load(f)
    
    embeddings_dict['original_posts'] = {
        str(k): v for k, v in embeddings_dict['original_posts'].items()
    }
    embeddings_dict['reply_posts'] = {
        str(k): v for k, v in embeddings_dict['reply_posts'].items()
    }
    
    posts_df = pd.read_csv(post_csv)
    comments_df = pd.read_csv(comment_csv)
    
    logger.info(f"✓ 加载了 {len(posts_df)} 条原始帖子")
    logger.info(f"✓ 加载了 {len(comments_df)} 条评论")
    
    reply_dict = defaultdict(list)
    for _, row in comments_df.iterrows():
        reply_to = str(row['in_reply_to_status_id_str'])
        reply_id = str(row['id_str'])
        if pd.notna(reply_to):
            reply_dict[reply_to].append(reply_id)
    
    logger.info(f"✓ 构建了 {len(reply_dict)} 个回复关系")
    
    post_data = {}
    successful = 0
    
    for idx, post_id in enumerate(posts_df['id_str'].astype(str)):
        tree_structure, chain_embeddings, depth_info = build_propagation_tree(
            post_id, reply_dict, embeddings_dict, max_depth=10
        )
        
        if chain_embeddings:
            chain_tensor = torch.tensor(chain_embeddings, dtype=torch.float32)
            depth_list = []
            
            id_to_idx = {id_: idx for idx, id_ in enumerate(tree_structure.keys())}
            indexed_structure = {}
            for id_, children in tree_structure.items():
                idx = id_to_idx[id_]
                indexed_structure[idx] = [
                    id_to_idx[child] for child in children 
                    if child in id_to_idx
                ]
            
            for i in range(len(chain_embeddings)):
                node_id = list(tree_structure.keys())[i]
                depth_list.append(depth_info[node_id])
            
            post_data[post_id] = {
                'chain': chain_tensor,
                'depth': depth_list,
                'structure': indexed_structure,
                'label': int(posts_df.loc[
                    posts_df['id_str'].astype(str) == post_id, 'is_rumor'
                ].values[0])
            }
            successful += 1
        
        if (idx + 1) % 100 == 0:
            logger.info(f"  处理进度: {idx+1}/{len(posts_df)}")
    
    logger.info(f"✓ 成功处理 {successful}/{len(posts_df)} 条帖子的传播链")
    logger.info("="*70)
    
    return post_data

def prepare_data_loaders(content_csv, text_emb_file, image_emb_file,
                        tweet_emb_file, post_csv, comment_csv,
                        batch_size=32):
    """准备数据加载器"""
    logger.info("\n" + "="*70)
    logger.info("准备数据加载器...")
    logger.info("="*70)
    
    # 1. 加载预计算的embeddings
    text_emb_dict, image_emb_dict, text_dim, image_dim = load_precomputed_embeddings(
        text_emb_file, image_emb_file
    )
    
    # 2. 加载content.csv并建立映射
    content_df = pd.read_csv(content_csv)
    
    mid_to_imgnum = {}
    mid_to_label = {}
    for _, row in content_df.iterrows():
        mid = str(row['mid'])
        imgnum = str(row['imgnum']) if pd.notna(row['imgnum']) else None
        label = int(row['label'])
        
        mid_to_label[mid] = label
        if imgnum:
            mid_to_imgnum[mid] = imgnum
    
    logger.info(f"✓ Content.csv: {len(mid_to_label)} 个样本")
    logger.info(f"✓ 图像映射: {len(mid_to_imgnum)} 个")
    logger.info(f"  标签分布: {Counter(mid_to_label.values())}")
    
    # 3. 加载传播链数据
    post_data = load_and_process_data(tweet_emb_file, post_csv, comment_csv)
    
    # 4. 构建完整数据字典
    complete_data = {}
    stats = {
        'total': len(mid_to_label),
        'has_text': 0,
        'has_imgnum': 0,
        'has_image': 0,
        'has_prop': 0,
        'complete': 0,
    }
    
    for mid in mid_to_label.keys():
        if mid not in text_emb_dict:
            continue
        stats['has_text'] += 1
        
        if mid not in mid_to_imgnum:
            continue
        stats['has_imgnum'] += 1
        
        imgnum = mid_to_imgnum[mid]
        if imgnum not in image_emb_dict:
            continue
        stats['has_image'] += 1
        
        if mid not in post_data:
            continue
        stats['has_prop'] += 1
        
        complete_data[mid] = {
            'text_embedding': text_emb_dict[mid],
            'image_embedding': image_emb_dict[imgnum],
            'label': mid_to_label[mid],
            'chain': post_data[mid]['chain'],
            'depth': post_data[mid]['depth'],
            'structure': post_data[mid]['structure'],
        }
        stats['complete'] += 1
    
    # 打印统计
    logger.info("\n" + "-"*70)
    logger.info("数据对齐统计:")
    logger.info(f"  总样本: {stats['total']}")
    logger.info(f"  有文本: {stats['has_text']} ({stats['has_text']/stats['total']*100:.1f}%)")
    logger.info(f"  有imgnum: {stats['has_imgnum']} ({stats['has_imgnum']/stats['total']*100:.1f}%)")
    logger.info(f"  有图像: {stats['has_image']} ({stats['has_image']/stats['total']*100:.1f}%)")
    logger.info(f"  有传播链: {stats['has_prop']} ({stats['has_prop']/stats['total']*100:.1f}%)")
    logger.info(f"  完整数据: {stats['complete']} ({stats['complete']/stats['total']*100:.1f}%)")
    logger.info("-"*70 + "\n")
    
    if stats['complete'] < 20:
        raise ValueError(f"完整数据太少（{stats['complete']}），无法训练")
    
    # 5. 划分数据集
    all_ids = list(complete_data.keys())
    all_labels = [complete_data[mid]['label'] for mid in all_ids]
    
    train_ids, temp_ids = train_test_split(
        all_ids, test_size=0.3, random_state=42, stratify=all_labels
    )
    
    temp_labels = [complete_data[mid]['label'] for mid in temp_ids]
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.67, random_state=42, stratify=temp_labels
    )
    
    logger.info("数据集划分:")
    logger.info(f"  训练集: {len(train_ids)} (标签: {Counter([complete_data[i]['label'] for i in train_ids])})")
    logger.info(f"  验证集: {len(val_ids)} (标签: {Counter([complete_data[i]['label'] for i in val_ids])})")
    logger.info(f"  测试集: {len(test_ids)} (标签: {Counter([complete_data[i]['label'] for i in test_ids])})")
    
    # 6. 保存数据集划分ID
    split_ids = {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
        'train_labels': [complete_data[mid]['label'] for mid in train_ids],
        'val_labels': [complete_data[mid]['label'] for mid in val_ids],
        'test_labels': [complete_data[mid]['label'] for mid in test_ids]
    }
    
    split_file = 'dataset_split_ids.json'
    with open(split_file, 'w') as f:
        json.dump(split_ids, f, indent=2)
    
    logger.info(f"✓ 数据集划分ID已保存到 {split_file}")
    logger.info("="*70 + "\n")
    
    # 7. 创建数据集
    train_dataset = RumorDataset({mid: complete_data[mid] for mid in train_ids})
    val_dataset = RumorDataset({mid: complete_data[mid] for mid in val_ids})
    test_dataset = RumorDataset({mid: complete_data[mid] for mid in test_ids})
    
    # 8. 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    
    prop_dim = complete_data[all_ids[0]]['chain'].size(-1)
    
    return train_loader, val_loader, test_loader, text_dim, image_dim, prop_dim

# ==================== 主训练函数 ====================
def train_model(args):
    """训练模型"""
    
    # 检查磁盘空间
    if not check_disk_space(min_gb=2):
        logger.warning("⚠️  磁盘空间不足,但继续训练...")
    
    logger.info("\n" + "="*70)
    logger.info("开始训练集成谣言检测模型")
    logger.info(f"Loss函数: 标准交叉熵 (CE)")
    logger.info(f"优化目标: F1 Macro")
    logger.info(f"学习率调度: ReduceLROnPlateau")
    logger.info("="*70)
    
    # 准备数据
    train_loader, val_loader, test_loader, text_dim, image_dim, prop_dim = prepare_data_loaders(
        args.content_csv, args.text_emb_file, args.image_emb_file,
        args.tweet_emb_file, args.post_csv, args.comment_csv, args.batch_size
    )
    
    logger.info(f"模型输入维度:")
    logger.info(f"  文本embedding: {text_dim}")
    logger.info(f"  图像embedding: {image_dim}")
    logger.info(f"  传播链embedding: {prop_dim}")
    logger.info(f"  批大小: {args.batch_size}")
    logger.info("="*70 + "\n")
    
    # 初始化模型
    model = IntegratedRumorDetectionModel(
        text_embedding_dim=text_dim,
        image_embedding_dim=image_dim,
        propagation_embedding_dim=prop_dim
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}\n")
    
    # 训练
    trainer = RumorTrainer(model, train_loader, val_loader, test_loader, device, args)
    
    best_val_f1 = 0
    best_epoch = 0
    no_improve = 0
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': []
    }
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f'\n{"="*70}')
        logger.info(f'Epoch {epoch}/{args.epochs}')
        logger.info(f'{"="*70}')
        
        # 训练
        train_loss, train_metrics = trainer.train_epoch(epoch)
        
        # 验证
        val_loss, val_metrics = trainer.evaluate(val_loader, phase="val", verbose=False)
        
        # 记录历史
        train_f1 = train_metrics['f1']
        val_f1 = val_metrics['f1']
        
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_f1'].append(train_f1)
        training_history['val_f1'].append(val_f1)
        
        # 简洁打印
        logger.info(f'\nEpoch {epoch} 汇总:')
        logger.info(f'  Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}')
        logger.info(f'  Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}')
        
        # 学习率调整
        trainer.scheduler.step(val_f1)
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_metrics': val_metrics,
                'args': vars(args)
            }
            
            save_path = 'best_model_ce_f1_macro.pth'
            if save_checkpoint_safely(checkpoint, save_path):
                logger.info(f'  ✓ 保存最佳模型 (F1: {best_val_f1:.4f})')
            else:
                logger.warning("  ⚠️  模型保存失败,但训练继续...")
        # else:
        #     no_improve += 1
        #     logger.info(f'  未改善 ({no_improve}/{args.early_stop_patience})')
        
        # 早停
        # if no_improve >= args.early_stop_patience:
        #     logger.info(f'\n早停触发！最佳epoch: {best_epoch}')
        #     break
    
    logger.info(f'\n{"="*70}')
    logger.info(f'训练完成！')
    logger.info(f'最佳验证F1: {best_val_f1:.4f} (epoch {best_epoch})')
    logger.info(f'{"="*70}')
    
    # 加载最佳模型进行测试
    save_path = 'best_model_ce_f1_macro.pth'
    try:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✓ 已加载最佳模型 (Epoch {checkpoint['epoch']})")
    except Exception as e:
        logger.warning(f"⚠️  无法加载最佳模型: {e}, 使用当前模型")
    
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test", verbose=True)
    
    # 保存结果
    results = {
        'args': vars(args),
        'best_epoch': best_epoch,
        'best_val_f1': float(best_val_f1),
        'test_metrics': test_metrics,
        'training_history': training_history
    }
    
    results_path = 'results_ce_f1_macro.json'
    try:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'\n✓ 结果已保存到 {results_path}')
    except Exception as e:
        logger.error(f"✗ 保存结果失败: {e}")
    
    return model, test_metrics, training_history

# ==================== 命令行参数解析 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='集成谣言检测模型训练 (标准CE Loss)')
    
    # 数据路径
    parser.add_argument('--content_csv', type=str, default='../content.csv')
    parser.add_argument('--text_emb_file', type=str, default='../../../autodl-tmp/pheme/original_text_embeddings.json')
    parser.add_argument('--image_emb_file', type=str, default='../../../autodl-tmp/pheme/image_embeddings.json')
    parser.add_argument('--tweet_emb_file', type=str, default='../../../autodl-tmp/pheme/tweet_embeddings.json')
    parser.add_argument('--post_csv', type=str, default='../../../autodl-tmp/pheme/post_tweet.csv')
    parser.add_argument('--comment_csv', type=str, default='../../../autodl-tmp/pheme/comment_tweet.csv')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    # parser.add_argument('--early_stop_patience', type=int, default=5, help='早停patience')
    
    return parser.parse_args()

# ==================== 主函数 ====================
if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    
    logger.info("\n" + "="*70)
    logger.info("运行配置:")
    logger.info("="*70)
    logger.info(f"  Loss函数: 标准交叉熵 (CE)")
    logger.info(f"  优化目标: F1 Macro")
    logger.info(f"  学习率调度: ReduceLROnPlateau")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  学习率: {args.learning_rate}")
    logger.info(f"  训练轮数: {args.epochs}")
    # logger.info(f"  早停patience: {args.early_stop_patience}")
    logger.info("="*70)
    
    try:
        model, metrics, history = train_model(args)
        
        logger.info('\n✅ 所有任务完成!')
        logger.info(f"\n最终测试结果:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        
    except KeyboardInterrupt:
        logger.info('\n⚠️  训练被用户中断')
        
    except Exception as e:
        logger.error(f'\n✗ 训练过程出错: {str(e)}')
        import traceback
        traceback.print_exc()
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info('\n✅ GPU缓存已清理')
