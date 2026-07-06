import os
import json
import random
import logging
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple
from collections import defaultdict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, ViTModel
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class CrossModalFusionModule(nn.Module):
    def __init__(self, text_image_dim, propagation_dim, fusion_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 投影层，将两种表征投影到相同维度
        self.text_image_projection = nn.Linear(text_image_dim, fusion_dim)
        self.propagation_projection = nn.Linear(propagation_dim, fusion_dim)
        
        # 交叉注意力层
        self.cross_attention1 = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attention2 = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True)
            
        # Layer Normalization
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.norm3 = nn.LayerNorm(fusion_dim)
        self.norm4 = nn.LayerNorm(fusion_dim)
        
        # Feed Forward Networks
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
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, text_image_repr, propagation_repr):
        # 投影到相同维度
        text_image = self.text_image_projection(text_image_repr)
        propagation = self.propagation_projection(propagation_repr)
        
    
    


        # 扩展维度以适应注意力机制
        text_image = text_image.unsqueeze(1)  # [batch_size, 1, fusion_dim]
        propagation = propagation.unsqueeze(1)  # [batch_size, 1, fusion_dim]
        
        # 交叉注意力: text_image -> propagation
        attn_output1, _ = self.cross_attention1(
            text_image, propagation, propagation)
        text_image = self.norm1(text_image + attn_output1)
        text_image = self.norm2(text_image + self.ffn1(text_image))
        
        # 交叉注意力: propagation -> text_image
        attn_output2, _ = self.cross_attention2(
            propagation, text_image, text_image)
        propagation = self.norm3(propagation + attn_output2)
        propagation = self.norm4(propagation + self.ffn2(propagation))
        
        # 压缩注意力维度
        text_image = text_image.squeeze(1)
        propagation = propagation.squeeze(1)
        
        # 连接并融合
        combined = torch.cat([text_image, propagation], dim=-1)
        fused = self.final_fusion(combined)
        
        return fused

class ContrastiveLearningModule(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def info_nce_loss(self, text_features, image_features):
        # 归一化特征
        text_features = F.normalize(text_features, dim=1)
        image_features = F.normalize(image_features, dim=1)
        
        # 计算相似度矩阵
        logits = torch.matmul(text_features, image_features.t()) / self.temperature
        
        # 正样本对在对角线上
        labels = torch.arange(logits.shape[0]).to(logits.device)
        
        # 计算对比损失
        loss = (F.cross_entropy(logits, labels) + 
                F.cross_entropy(logits.t(), labels)) / 2
        return loss

class ModalityDisentanglement(nn.Module):
    def __init__(self, feature_dim, shared_dim):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(feature_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )
        
        self.specific_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim - shared_dim),
            nn.ReLU(),
            nn.Linear(feature_dim - shared_dim, feature_dim - shared_dim)
        )
        
        self.modality_discriminator = nn.Sequential(
            nn.Linear(shared_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, text_features, image_features):
        # 提取共享特征
        text_shared = self.shared_encoder(text_features)
        image_shared = self.shared_encoder(image_features)
        
        # 提取特定特征
        text_specific = self.specific_encoder(text_features)
        image_specific = self.specific_encoder(image_features)
        
        # 对抗训练
        modality_pred_text = self.modality_discriminator(text_shared)
        modality_pred_image = self.modality_discriminator(image_shared)
        adv_loss = F.binary_cross_entropy_with_logits(
            modality_pred_text, 
            torch.zeros_like(modality_pred_text)
        ) + F.binary_cross_entropy_with_logits(
            modality_pred_image,
            torch.ones_like(modality_pred_image)
        )
        
        # 组合特征
        text_combined = torch.cat([text_shared, text_specific], dim=-1)
        image_combined = torch.cat([image_shared, image_specific], dim=-1)
        
        return text_combined, image_combined, adv_loss

class GuidedAttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, 8)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, text_features, image_features):
        # 互注意力
        attended_text, _ = self.attention(
            text_features, image_features, image_features
        )
        
        # 动态门控
        gate = self.gate(torch.cat([text_features, attended_text], dim=-1))
        enhanced_features = gate * attended_text + (1 - gate) * text_features
        return enhanced_features

class MultiViewConsistency(nn.Module):
    def __init__(self, aug_transforms):
        super().__init__()
        self.aug_transforms = aug_transforms
        
    def forward(self, images):
        # 生成多个增强视图
        views = [self.aug_transforms(img) for img in images]
        return views


class MultiModalFusion(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int, shared_dim: int):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        
        # Co-attention layers
        self.text_image_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.image_text_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        
        # 模态解耦模块
        self.disentanglement = ModalityDisentanglement(hidden_dim, shared_dim)
        
        # 引导注意力模块
        self.guided_attention = GuidedAttentionModule(hidden_dim)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor,
                text_attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 投影到相同维度
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        
        # 计算注意力掩码
        attention_mask = text_attention_mask.float()
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))
        
        # 双向注意力
        text_attended, _ = self.text_image_attention(text_proj, image_proj, image_proj)
        image_attended, _ = self.image_text_attention(image_proj, text_proj, text_proj, 
                                                    key_padding_mask=attention_mask)
        
        # 获取[CLS]位置的特征
        text_cls = text_attended[:, 0]
        image_cls = image_attended.mean(dim=1)
        
        # 模态解耦
        text_combined, image_combined, adv_loss = self.disentanglement(text_cls, image_cls)
        
        # 引导注意力增强
        enhanced_text = self.guided_attention(text_combined.unsqueeze(1), image_combined.unsqueeze(1)).squeeze(1)
        
        # 融合特征
        fused_features = self.fusion_layer(torch.cat([enhanced_text, image_combined], dim=1))
        return fused_features, adv_loss

class IntegratedRumorDetectionModel(nn.Module):
    def __init__(self, hidden_size, num_classes=2):
        super().__init__()
        # 文本-图像模型组件
        self.text_encoder = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')

        
        # 获取维度
        text_dim = self.text_encoder.config.hidden_size
        image_dim = self.image_encoder.config.hidden_size
        hidden_dim = 512
        shared_dim = 256
        
        # 文本-图像模型的组件
        self.text_image_fusion = MultiModalFusion(text_dim, image_dim, hidden_dim, shared_dim)
        self.contrastive = ContrastiveLearningModule()
        
        # 传播链模型组件
        self.transformer_layers = nn.ModuleList([
            TreeTransformerLayer(hidden_size, num_heads=8, dropout=0.1)
            for _ in range(3)
        ])
        
        # 交叉模态融合模块
        self.cross_fusion = CrossModalFusionModule(
            text_image_dim=hidden_dim,
            propagation_dim=hidden_size,
            fusion_dim=hidden_dim
        )
        
        # 最终分类器
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
        
    
    def forward(self, input_ids, attention_mask, image, chain_embeddings, depth_info, chain_structure):
        
        
        # 1. 检查文本编码
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_features = text_outputs.last_hidden_state
        
        # 2. 检查图像编码
        image_outputs = self.image_encoder(image)
        image_features = image_outputs.last_hidden_state
        
        # 3. 检查对比损失
        contrastive_loss = self.contrastive.info_nce_loss(
            text_features[:, 0], 
            image_features.mean(dim=1)
        )
        
        
        # 4. 检查融合特征
        text_image_features, adv_loss = self.text_image_fusion(
            text_features, image_features, attention_mask)
        
        
        # 2. 传播链表征
        batch_size, max_length, _ = chain_embeddings.size()
        
        # 创建深度衰减掩码
        depth_masks = torch.zeros((batch_size, max_length)).to(chain_embeddings.device)
        for i, depths in enumerate(depth_info):
            for j, depth in enumerate(depths):
                if j < len(depths):
                    depth_masks[i][j] = torch.exp(-torch.tensor(depth).float())
        
        # 创建传播链结构掩码
        chain_mask = torch.zeros((batch_size, max_length, max_length)).to(chain_embeddings.device)
        for i, structure in enumerate(chain_structure):
            for parent, children in structure.items():
                parent_idx = list(structure.keys()).index(parent)
                chain_mask[i][parent_idx][parent_idx] = 1
                for child in children:
                    child_idx = list(structure.keys()).index(child)
                    chain_mask[i][parent_idx][child_idx] = 1
                    chain_mask[i][child_idx][parent_idx] = 1
        
        # Transformer处理
        x = chain_embeddings
        for layer in self.transformer_layers:
            x = layer(x, depth_masks, chain_mask)
        
        # 获取根节点表征
        propagation_features = x[:, 0, :]
        
        # 3. 交叉模态融合
        fused_features = self.cross_fusion(text_image_features, propagation_features)
        
        # 4. 分类
        logits = self.classifier(fused_features)
        
        return logits, contrastive_loss, adv_loss

class IntegratedRumorTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, contrastive_weight=0.1, adv_weight=0.01):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.contrastive_weight = contrastive_weight
        self.adv_weight = adv_weight
        
        # 优化器
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_labels = []
        all_predictions = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            
            # 处理文本-图像数据
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images = batch['image'].to(self.device)
            
            # 处理传播链数据
            chains = batch['chains'].to(self.device)
            depths = batch['depths']
            structures = batch['structures']
            
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            logits, contrastive_loss, adv_loss = self.model(
                input_ids, attention_mask, images,
                chains, depths, structures
            )
            # 检查logits
            # print(f"Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
            
            # 计算总损失
            cls_loss = self.criterion(logits, labels)

            

            
            total_loss = (cls_loss + 
                         self.contrastive_weight * contrastive_loss +
                         self.adv_weight * adv_loss)
            
            
            
            
            
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            _, predicted = logits.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Batch [{batch_idx+1}/{len(self.train_loader)}], '
                      f'Loss: {total_loss.item():.4f}, '
                      f'Cls Loss: {cls_loss.item():.4f}, '
                      f'Con Loss: {contrastive_loss.item():.4f}, '
                      f'Adv Loss: {adv_loss.item():.4f}')
        
        metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions))
        return total_loss / len(self.train_loader), metrics
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                chains = batch['chains'].to(self.device)
                depths = batch['depths']
                structures = batch['structures']
                labels = batch['label'].to(self.device)
                
                logits, contrastive_loss, adv_loss = self.model(
                    input_ids, attention_mask, images,
                    chains, depths, structures
                )
                
                cls_loss = self.criterion(logits, labels)
                batch_loss = (cls_loss + 
                            self.contrastive_weight * contrastive_loss +
                            self.adv_weight * adv_loss)
                
                total_loss += batch_loss.item()
                
                _, predicted = logits.max(1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions))
        return total_loss / len(data_loader), metrics

class IntegratedRumorDataset(Dataset):
    def __init__(self, text_list, image_paths, chains, depths, structures, labels, 
                 tokenizer, transform, max_length=128, augment=False):
        self.texts = text_list
        self.image_paths = image_paths
        self.chains = chains
        self.depths = depths
        self.structures = structures
        self.labels = labels
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.augment = augment
        
        # 文本增强策略
        self.text_aug_methods = [
            self.random_deletion,
            self.random_swap,
        ]
        
        # 预处理structures，将ID映射到索引
        self.processed_structures = []
        for struct in structures:
            processed = {}
            id_to_idx = {id_: idx for idx, id_ in enumerate(struct.keys())}
            for id_, children in struct.items():
                idx = id_to_idx[id_]
                processed[idx] = [id_to_idx[child] for child in children]
            self.processed_structures.append(processed)
    
    def random_deletion(self, text, p=0.1):
        words = text.split()
        words = [word for word in words if random.random() > p]
        return ' '.join(words)
        
    def random_swap(self, text, n=1):
        words = text.split()
        for _ in range(n):
            if len(words) >= 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 处理文本-图像部分
        text = str(self.texts[idx])
        
        # 文本增强
        if self.augment:
            aug_method = random.choice(self.text_aug_methods)
            text = aug_method(text)
            
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # print(f"Sample {idx} input_ids shape: {encoding['input_ids'].shape}")
        
                
        # 处理图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        # print(f"Sample {idx} image shape: {image.shape}")

        
        # 处理传播链数据
        chain = self.chains[idx]
        # print(f"Sample {idx} chain shape: {chain.shape}")

        depth = self.depths[idx]
        structure = self.processed_structures[idx]
        
        # 获取标签
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'image': image,
            'chains': chain,
            'depths': depth,
            'structures': structure,
            'label': label
        }

def integrated_collate_fn(batch):
    """
    自定义的collate函数，处理变长序列和特殊的数据结构
    """
    # 收集每个模态的数据
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    images = torch.stack([item['image'] for item in batch])
    chains = [item['chains'] for item in batch]
    depths = [item['depths'] for item in batch]
    structures = [item['structures'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # 处理传播链数据
    max_len = max(chain.size(0) for chain in chains)
    padded_chains = pad_chains(chains, max_len)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'image': images,
        'chains': padded_chains,
        'depths': depths,
        'structures': structures,
        'label': labels
    }

def pad_chains(chains, max_len=None):
    if max_len is None:
        max_len = max(chain.size(0) for chain in chains)
    
    padded_chains = []
    for chain in chains:
        if chain.size(0) < max_len:
            padding = torch.zeros(max_len - chain.size(0), chain.size(1))
            padded_chain = torch.cat([chain, padding], dim=0)
            padded_chains.append(padded_chain)
        else:
            padded_chains.append(chain)
    
    return torch.stack(padded_chains)

class LayerDecayAttention(nn.Module):
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
        
        # 1. 线性变换和重塑
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. 计算原始注意力分数
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        
        # 3. 应用深度掩码
        depth_masks = depth_masks.unsqueeze(1).unsqueeze(2)
        attention_scores = attention_scores * depth_masks
        
        
        # 4. 应用链掩码（如果存在）
        if chain_mask is not None:
            # 确保每一行至少有一个有效值，转换为布尔类型
            row_has_valid = (attention_scores != float('-inf')).any(dim=-1, keepdim=True)
            # 创建对角线掩码并转换为与attention_scores相同的类型
            diag_mask = torch.eye(attention_scores.size(-1), device=attention_scores.device)[None, None]
            # 计算掩码，确保类型正确
            mask_for_empty = (~row_has_valid).float() * diag_mask
            # 使用掩码填充值
            attention_scores = attention_scores.masked_fill(mask_for_empty.bool(), 0.0)
        
        # 应用softmax
        attn = torch.softmax(attention_scores, dim=-1)
        if torch.isnan(attn).any():
            print(f"Warning: NaN values detected after softmax: {torch.isnan(attn).sum().item()}")
        
        # 6. Dropout和注意力应用
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        
        # 7. 重塑和最终投影
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_length, self.hidden_size)
        out = self.out(out)
        
        if torch.isnan(out).any():
            print("Warning: NaN values in output")
        
        return out


class TreeTransformerLayer(nn.Module):
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

def prepare_integrated_data(
    text_list: List[str],
    image_paths: List[str],
    chains: List[torch.Tensor],
    depths: List[List[int]],
    structures: List[Dict],
    labels: List[int],
    tokenizer_name: str = 'hfl/chinese-roberta-wwm-ext',
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    # 基础图像转换
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 增强图像转换
    aug_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 数据集划分 (70% 训练, 30% 用于验证和测试)
    train_texts, temp_texts, train_images, temp_images, \
    train_chains, temp_chains, train_depths, temp_depths, \
    train_structures, temp_structures, train_labels, temp_labels = train_test_split(
        text_list, image_paths, chains, depths, structures, labels,
        test_size=0.3, random_state=42
    )
    
    # 将剩余数据划分为验证集和测试集 (2:1)
    val_texts, test_texts, val_images, test_images, \
    val_chains, test_chains, val_depths, test_depths, \
    val_structures, test_structures, val_labels, test_labels = train_test_split(
        temp_texts, temp_images, temp_chains, temp_depths, 
        temp_structures, temp_labels,
        test_size=0.66, random_state=42
    )
    
    # 创建数据集
    train_dataset = IntegratedRumorDataset(
        train_texts, train_images, train_chains, train_depths, train_structures, train_labels,
        tokenizer, aug_transform, augment=True
    )
    val_dataset = IntegratedRumorDataset(
        val_texts, val_images, val_chains, val_depths, val_structures, val_labels,
        tokenizer, base_transform, augment=False
    )
    test_dataset = IntegratedRumorDataset(
        test_texts, test_images, test_chains, test_depths, test_structures, test_labels,
        tokenizer, base_transform, augment=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=integrated_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=integrated_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=integrated_collate_fn
    )
    
    return train_loader, val_loader, test_loader

def train_integrated_model(
    content_csv_path: str,
    image_folder: str,
    embeddings_file: str,
    post_csv: str,
    comment_csv: str,
    num_epochs: int = 20,
    device: str = 'cuda',
    contrastive_weight: float = 0.1,
    adv_weight: float = 0.01
) -> nn.Module:
    # 1. 加载文本-图像数据
    text_list, image_paths, text_image_labels, post_ids = load_data(content_csv_path, image_folder)
    
    # 2. 加载传播链数据
    post_data = load_and_process_data(embeddings_file, post_csv, comment_csv)
    aligned_chains = []
    aligned_depths = []
    aligned_structures = []
    aligned_labels = []
    aligned_texts = []
    aligned_images = []
    
    for idx, post_id in enumerate(post_ids):
        if post_id in post_data:
            data = post_data[post_id]
            aligned_chains.append(data['chain'])
            aligned_depths.append(data['depth'])
            aligned_structures.append(data['structure'])
            aligned_labels.append(data['label'])
            aligned_texts.append(text_list[idx])
            aligned_images.append(image_paths[idx])
    
    logger.info(f"对齐后的数据数量: {len(aligned_chains)}")
    
    # 4. 准备数据加载器
    train_loader, val_loader, test_loader = prepare_integrated_data(
        aligned_texts, aligned_images, aligned_chains, 
        aligned_depths, aligned_structures, aligned_labels
    )
    # 5. 初始化模型
    hidden_size = aligned_chains[0].size(-1)  # 使用embedding维度作为隐藏层大小
    model = IntegratedRumorDetectionModel(hidden_size=hidden_size)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 6. 初始化训练器
    trainer = IntegratedRumorTrainer(
        model, train_loader, val_loader, test_loader, device,
        contrastive_weight=contrastive_weight,
        adv_weight=adv_weight
    )
    
    # 7. 训练循环
    best_val_f1 = 0
    best_epoch = 0
    patience = 5  # 早停的耐心值
    no_improve = 0
    
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        
        # 训练
        train_loss, train_metrics = trainer.train_epoch()
        
        # 验证
        val_loss, val_metrics = trainer.evaluate(val_loader)
        
        # 打印结果
        logger.info(f'Training Loss: {train_loss:.4f}')
        logger.info(f'Training Metrics: {train_metrics}')
        logger.info(f'Validation Loss: {val_loss:.4f}')
        logger.info(f'Validation Metrics: {val_metrics}')
        
        # 更新学习率
        trainer.scheduler.step(val_metrics['f1'])
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), 'best_integrated_model.pth')
            logger.info(f'Saved new best model with F1: {best_val_f1:.4f}')
        else:
            no_improve += 1
        
        # 早停检查
        # if no_improve >= patience:
        #     logger.info(f'Early stopping triggered after epoch {epoch+1}')
        #     break
            
        logger.info('-' * 60)
    
    logger.info(f'Best validation F1: {best_val_f1:.4f} at epoch {best_epoch+1}')
    
    # 8. 测试最佳模型
    model.load_state_dict(torch.load('best_integrated_model.pth'))
    test_loss, test_metrics = trainer.evaluate(test_loader)
    
    logger.info('Final Test Results:')
    logger.info(f'Test Loss: {test_loss:.4f}')
    logger.info(f'Test Metrics: {test_metrics}')
    
    return model, test_metrics



def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    Returns:
        包含accuracy、precision、recall、f1四个指标的字典
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, 
        y_pred, 
        average='binary',  # 二分类任务使用binary
        zero_division=0    # 处理分母为0的情况
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
    
    # 确保所有指标都是有效数值
    for k, v in metrics.items():
        if np.isnan(v) or np.isinf(v):
            logger.warning(f'发现无效的评估指标 {k}: {v}, 将其设为0')
            metrics[k] = 0.0
            
    return metrics

def build_propagation_tree(post_id, reply_dict, embeddings_dict, visited=None, max_chain_length=50):
    """
    添加最大链长度限制
    """
    if visited is None:
        visited = set()
    
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
        
        # 限制传播链长度
        if len(embeddings) >= max_chain_length:
            return tree, embeddings[:max_chain_length], depths
        
        for reply_id in replies:
            sub_tree, sub_embeddings, sub_depths = build_propagation_tree(
                reply_id, reply_dict, embeddings_dict, visited, max_chain_length
            )
            if sub_tree:
                tree[post_id].append(reply_id)
                tree.update(sub_tree)
                embeddings.extend(sub_embeddings)
                
                for node, depth in sub_depths.items():
                    depths[node] = depth + 1
                    
                # 检查是否超过最大长度
                if len(embeddings) >= max_chain_length:
                    break
        
        return tree, embeddings[:max_chain_length], depths
        
    except Exception as e:
        print(f"Error processing post_id {post_id}: {str(e)}")
        return {}, [], {}
    
    replies = reply_dict.get(post_id, [])
    
    tree = {post_id: []}
    embeddings = [current_embedding]
    depths = {post_id: 0}
    
    for reply_id in replies:
        sub_tree, sub_embeddings, sub_depths = build_propagation_tree(
            reply_id, reply_dict, embeddings_dict, visited
        )
        if sub_tree:
            tree[post_id].append(reply_id)
            tree.update(sub_tree)
            embeddings.extend(sub_embeddings)
            
            for node, depth in sub_depths.items():
                depths[node] = depth + 1
    
    return tree, embeddings, depths

def load_data(csv_path: str, image_folder: str) -> Tuple[List[str], List[str], List[int]]:
    """加载数据并匹配图像路径"""
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"图片文件夹不存在: {image_folder}")
        
    # 读取CSV文件
    logger.info(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"CSV文件中的总行数: {len(df)}")
    
    # 数据检查
    logger.info("\n=== 数据检查 ===")
    logger.info(f"列名: {df.columns.tolist()}")
    logger.info("\n数据类型统计:")
    logger.info(df.dtypes)
    logger.info("\n空值统计:")
    logger.info(df.isnull().sum())
    
    # 检查文本数据
    text_lengths = df['text'].str.len()
    logger.info("\n文本长度统计:")
    logger.info(f"最小长度: {text_lengths.min()}")
    logger.info(f"最大长度: {text_lengths.max()}")
    logger.info(f"平均长度: {text_lengths.mean():.2f}")
    logger.info(f"空文本数量: {len(df[df['text'].isna()])}")
    logger.info(f"空字符串数量: {len(df[df['text'] == ''])}")
    
    # 检查标签分布
    logger.info("\n标签分布:")
    label_dist = df['label'].value_counts()
    logger.info(label_dist)
    logger.info(f"标签值集合: {sorted(df['label'].unique())}")
    
    # 异常值检查
    if 'imgnum' in df.columns:
        logger.info("\n图像编号检查:")
        logger.info(f"空的图像编号: {len(df[df['imgnum'].isna()])}")
        logger.info(f"图像编号范围: {df['imgnum'].min()} - {df['imgnum'].max()}")
    
    texts = []
    image_paths = []
    labels = []
    post_ids = []
    missing_images = []
    invalid_labels = []
    empty_texts = []
    
    for idx, row in df.iterrows():
        try:
            # 检查文本
            if pd.isna(row['text']) or str(row['text']).strip() == '':
                empty_texts.append(idx)
                continue
                
            # 检查标签
            if pd.isna(row['label']) or int(row['label']) not in [0, 1]:
                invalid_labels.append(idx)
                continue
                
            img_path = os.path.join(image_folder, f"{str(row['imgnum'])}.jpg")
            # 检查图像文件是否存在
            if os.path.exists(img_path):
                texts.append(str(row['text']))
                image_paths.append(img_path)
                labels.append(int(row['label']))
                post_ids.append(str(row['mid']))
            else:
                missing_images.append(row['imgnum'])
        except Exception as e:
            logger.error(f"处理第 {idx} 行时出错: {str(e)}")
            logger.error(f"问题数据: {row}")
            
    # 打印统计信息
    logger.info("\n=== 数据处理统计 ===")
    logger.info(f"总样本数: {len(df)}")
    logger.info(f"有效样本数: {len(texts)}")
    logger.info(f"缺失图片数: {len(missing_images)}")
    logger.info(f"无效标签数: {len(invalid_labels)}")
    logger.info(f"空文本数: {len(empty_texts)}")
    
    if len(texts) == 0:
        raise ValueError("没有找到有效的样本！请检查数据路径和文件格式是否正确。")
        
    return texts, image_paths, labels, post_ids

def load_and_process_data(embeddings_file, post_csv, comment_csv):
    """加载并处理数据，返回处理后的数据和标签"""
    print("\n=== 加载数据文件 ===")
    # 加载embeddings
    
    with open(embeddings_file, 'r') as f:
        embeddings_dict = json.load(f)
    
    # 检查embeddings
    
    
    # 随机抽样检查embedding维度
    orig_sample = next(iter(embeddings_dict['original_posts'].values()))
    reply_sample = next(iter(embeddings_dict['reply_posts'].values()))
    
    # 读取CSV文件
    print("\n读取CSV文件:")
    posts_df = pd.read_csv(post_csv)
    comments_df = pd.read_csv(comment_csv)
    
   
    
    # 构建回复关系字典
    reply_dict = defaultdict(list)
    invalid_replies = 0
    for _, row in comments_df.iterrows():
        try:
            reply_to = str(row['in_reply_to_status_id_str'])
            reply_id = str(row['mid'])
            if pd.notna(reply_to):
                reply_dict[reply_to].append(reply_id)
        except Exception as e:
            invalid_replies += 1
            
    
    
    post_data = {}
    invalid_trees = 0
    empty_chains = 0
    
    for idx, post_id in enumerate(posts_df['id_str'].astype(str)):
        try:
            tree_structure, chain_embeddings, depth_info = build_propagation_tree(
                post_id, reply_dict, embeddings_dict
            )
            
            if not tree_structure:
                invalid_trees += 1
                continue
                
            if not chain_embeddings:
                empty_chains += 1
                continue
                
            chain_tensor = torch.tensor(chain_embeddings)
            depth_list = []
            for i, _ in enumerate(chain_embeddings):
                node_id = list(tree_structure.keys())[i]
                depth_list.append(depth_info[node_id])
            
            post_data[post_id] = {
                'chain': chain_tensor,
                'depth': depth_list,
                'structure': tree_structure,
                'label': int(posts_df.loc[idx, 'is_rumor'])
            }
            
            # 定期打印进度
            if (idx + 1) % 100 == 0:
                print(f"已处理 {idx + 1}/{len(posts_df)} 条帖子")
                
        except Exception as e:
            print(f"处理帖子 {post_id} 时出错: {str(e)}")
    
    # 最终统计
    print("\n=== 最终数据统计 ===")
    print(f"成功处理的帖子数: {len(post_data)}")
    print(f"无效树结构数: {invalid_trees}")
    print(f"空传播链数: {empty_chains}")
    
    # 检查传播链长度分布
    chain_lengths = [len(data['chain']) for data in post_data.values()]
    print("\n传播链长度统计:")
    print(f"最短链长度: {min(chain_lengths)}")
    print(f"最长链长度: {max(chain_lengths)}")
    print(f"平均链长度: {sum(chain_lengths)/len(chain_lengths):.2f}")
    
    # 检查标签分布
    labels = [data['label'] for data in post_data.values()]
    print("\n标签分布:")
    label_counts = pd.Series(labels).value_counts()
    print(label_counts)
    
    return post_data

if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设置参数
    content_csv_path = 'weibo_content.csv'
    image_folder = os.path.join('weibo_images', 'weibo_images_all')
    embeddings_file = 'weibo_embeddings.json'
    post_csv = 'weibo_ori.csv'
    comment_csv = 'weibo_reply.csv'
    
    num_epochs = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    contrastive_weight = 0.1
    adv_weight = 0.01
    
    try:
        # 训练模型
        logger.info('Starting integrated model training...')
        model, metrics = train_integrated_model(
            content_csv_path=content_csv_path,
            image_folder=image_folder,
            embeddings_file=embeddings_file,
            post_csv=post_csv,
            comment_csv=comment_csv,
            num_epochs=num_epochs,
            device=device,
            contrastive_weight=contrastive_weight,
            adv_weight=adv_weight
        )
        
        # 保存结果
        results = {
            'test_metrics': metrics,
            'training_params': {
                'num_epochs': num_epochs,
                'device': device,
                'contrastive_weight': contrastive_weight,
                'adv_weight': adv_weight
            }
        }
        
        # 使用pandas将结果保存为CSV
        results_df = pd.DataFrame([metrics])
        results_df.to_csv('integrated_test_results.csv', index=False)
        
        logger.info('Training completed successfully!')
        logger.info(f'Final test metrics: {metrics}')
        
    except Exception as e:
        logger.error(f'An error occurred during training: {str(e)}')
        raise
