import os
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from typing import Dict, List
import pickle
import torch.nn.functional as F

# Transformers for text embedding
from transformers import AutoModel, AutoTokenizer

# For image embedding
import torchvision.transforms as transforms
from torchvision import models

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingExtractor:
    def __init__(self, 
                 qwen_model_path='../../../autodl-tmp/qwen3',
                 device='cuda',
                 batch_size=16):
        """
        初始化embedding提取器
        
        Args:
            qwen_model_path: Qwen3-Embedding-8B模型本地路径
            dinov2_model_path: DINOv2模型本地路径
            device: 计算设备
            batch_size: 批处理大小
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        # 初始化属性
        self.text_tokenizer = None
        self.text_model = None
        self.default_task = None
        self.image_model = None
        
        logger.info(f"使用设备: {self.device}")
        logger.info("正在加载模型...")
        
        # 加载文本模型
        self.load_text_model(qwen_model_path)
        
        # 加载图像模型
        self.load_image_model()
        
        logger.info("模型加载完成!")
    
    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        根据官网推荐的方式提取最后一个token的embedding
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """
        根据官网推荐格式生成指令
        """
        return f'Instruct: {task_description}\nQuery: {query}'

    def load_text_model(self, model_path):
        """加载Qwen3-Embedding-8B文本模型"""
        try:
            # 检查transformers版本
            import transformers
            transformers_version = transformers.__version__
            logger.info(f"当前transformers版本: {transformers_version}")
            
            # 检查版本是否满足要求
            try:
                from packaging import version
                if version.parse(transformers_version) < version.parse("4.51.0"):
                    logger.error(f"transformers版本 {transformers_version} 不支持Qwen3，必须>=4.51.0")
                    logger.info("请升级transformers: pip install transformers>=4.51.0 --upgrade")
                    raise Exception("transformers版本不兼容")
            except ImportError:
                logger.warning("无法检查transformers版本，继续尝试加载")
            
            # 检查模型路径是否存在
            if not os.path.exists(model_path):
                raise Exception(f"模型路径不存在: {model_path}")
            
            required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
            for file in required_files:
                file_path = os.path.join(model_path, file)
                if not os.path.exists(file_path):
                    logger.warning(f"缺少必要文件: {file_path}")
            
            # 加载tokenizer
            logger.info("加载tokenizer...")
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side='left',  # 官网推荐
                trust_remote_code=True
            )
            logger.info("tokenizer加载成功")
            
            # 加载模型
            logger.info("加载Qwen3-Embedding模型...")
            try:
                # 首选配置：使用flash_attention_2
                self.text_model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    device_map="auto"
                ).to(self.device)
                logger.info("使用flash_attention_2加载成功")
            except Exception as e1:
                logger.warning(f"flash_attention_2加载失败，尝试标准配置: {e1}")
                try:
                    # 备选配置：标准配置
                    self.text_model = AutoModel.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
                    ).to(self.device)
                    logger.info("使用标准配置加载成功")
                except Exception as e2:
                    logger.warning(f"标准配置加载失败，尝试基础配置: {e2}")
                    # 最后尝试：基础配置
                    self.text_model = AutoModel.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    ).to(self.device)
                    logger.info("使用基础配置加载成功")
            
            self.text_model.eval()
            
            # 获取embedding维度
            embedding_dim = getattr(self.text_model.config, 'hidden_size', 'unknown')
            logger.info(f"Qwen3-Embedding模型加载成功，embedding维度: {embedding_dim}")
            
            # 设置默认任务描述
            self.default_task = 'Given a social media post, generate embedding for content analysis and similarity matching'
            
        except Exception as e:
            logger.error(f"加载Qwen3-Embedding模型失败: {e}")
            logger.error("请检查以下事项:")
            logger.error("1. transformers版本是否>=4.51.0")
            logger.error("2. 模型路径是否正确且包含所有必要文件")
            logger.error("3. 模型是否为Qwen3-Embedding-8B")
            logger.error("4. 是否有足够的显存/内存")
            raise Exception(f"无法加载Qwen3-Embedding模型: {e}")
            
    def load_image_model(self):
        """加载图像embedding模型 - 使用ResNet50"""
        try:
            logger.info("加载ResNet50图像模型...")
            
            # 加载预训练的ResNet50模型
            self.image_model = models.resnet50(pretrained=True)
            
            # 移除最后的全连接层，保留特征提取层
            self.image_model = torch.nn.Sequential(*list(self.image_model.children())[:-1])
            
            # 将模型移动到指定设备
            self.image_model.to(self.device)
            
            # 设置为评估模式
            self.image_model.eval()
            
            # 设置模型名称和embedding维度
            self.image_model_name = "ResNet50"
            self.image_embedding_dim = 2048
            
            # 定义图像变换
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整图像大小
                transforms.ToTensor(),  # 转换为张量
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet的均值
                    std=[0.229, 0.224, 0.225]    # ImageNet的标准差
                )
            ])
            
            logger.info(f"{self.image_model_name}模型加载成功，特征维度: {self.image_embedding_dim}")
            
        except Exception as e:
            logger.error(f"加载ResNet50模型失败: {e}")
            raise Exception(f"无法加载ResNet50模型: {e}")
    
    def extract_text_embeddings(self, texts: List[str], ids: List[str], custom_task: str = None) -> Dict:
        """
        使用Qwen3-Embedding批量提取文本embeddings
        
        Args:
            texts: 文本列表
            ids: 对应的ID列表
            custom_task: 自定义任务描述
            
        Returns:
            字典：{id: embedding_vector}
        """
        logger.info(f"开始提取{len(texts)}条文本的embeddings...")
        
        # 检查模型是否加载成功
        if self.text_model is None or self.text_tokenizer is None:
            logger.error("Qwen3-Embedding模型未成功加载，无法提取文本embeddings")
            raise Exception("文本模型未加载")
        
        embeddings_dict = {}
        task_description = custom_task or self.default_task
        
        # 分批处理
        for i in tqdm(range(0, len(texts), self.batch_size), desc="提取文本embeddings"):
            batch_texts = texts[i:i+self.batch_size]
            batch_ids = ids[i:i+self.batch_size]
            
            try:
                with torch.no_grad():
                    # 根据官网文档，为文本添加指令格式
                    formatted_texts = []
                    for text in batch_texts:
                        # 清理文本
                        text = str(text).strip()
                        if not text:
                            text = "[empty]"
                        formatted_text = self.get_detailed_instruct(task_description, text)
                        formatted_texts.append(formatted_text)
                    
                    # 根据官网推荐设置max_length=8192
                    max_length = 8192
                    
                    # Tokenize
                    batch_dict = self.text_tokenizer(
                        formatted_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    batch_dict = batch_dict.to(self.device)
                    
                    # 前向传播
                    outputs = self.text_model(**batch_dict)
                    
                    # 使用官网推荐的last_token_pool方法
                    embeddings = self.last_token_pool(
                        outputs.last_hidden_state, 
                        batch_dict['attention_mask']
                    )
                    
                    # 归一化embeddings（官网推荐）
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                    # 转为CPU numpy数组存储
                    embeddings = embeddings.cpu().numpy()
                    
                    for j, text_id in enumerate(batch_ids):
                        embeddings_dict[str(text_id)] = embeddings[j]
                        
            except Exception as e:
                logger.error(f"处理文本批次 {i//self.batch_size + 1} 时出错: {e}")
                
                # 逐条处理失败的批次
                for j, (text, text_id) in enumerate(zip(batch_texts, batch_ids)):
                    try:
                        text = str(text).strip()
                        if not text:
                            text = "[empty]"
                            
                        with torch.no_grad():
                            formatted_text = self.get_detailed_instruct(task_description, text)
                            
                            batch_dict = self.text_tokenizer(
                                [formatted_text],
                                padding=True,
                                truncation=True,
                                max_length=8192,
                                return_tensors="pt",
                            ).to(self.device)
                            
                            outputs = self.text_model(**batch_dict)
                            embedding = self.last_token_pool(
                                outputs.last_hidden_state, 
                                batch_dict['attention_mask']
                            )[0]
                            embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1)[0]
                            embedding = embedding.cpu().numpy()
                                
                            embeddings_dict[str(text_id)] = embedding
                            
                    except Exception as e2:
                        logger.error(f"处理单条文本 {text_id} 失败: {e2}")
                        # 使用零向量作为fallback
                        embeddings_dict[str(text_id)] = np.zeros(4096)  # Qwen3-Embedding-8B维度
        
        logger.info(f"文本embedding提取完成，成功提取 {len(embeddings_dict)} 条")
        return embeddings_dict
    
    def extract_image_embeddings(self, image_paths: List[str], ids: List[str]) -> Dict:
        """
        批量提取图像embeddings
        
        Args:
            image_paths: 图像路径列表
            ids: 对应的ID列表
            
        Returns:
            字典：{id: embedding_vector}
        """
        logger.info(f"开始提取{len(image_paths)}张图像的embeddings...")
        
        embeddings_dict = {}
        
        # 分批处理
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="提取图像embeddings"):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_ids = ids[i:i+self.batch_size]
            
            batch_images = []
            valid_indices = []
            
            # 加载批次中的图像
            for j, img_path in enumerate(batch_paths):
                try:
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = self.image_transform(img)
                        batch_images.append(img_tensor)
                        valid_indices.append(j)
                    else:
                        logger.warning(f"图像文件不存在: {img_path}")
                        
                except Exception as e:
                    logger.error(f"加载图像 {img_path} 失败: {e}")
            
            if batch_images:
                try:
                    # 堆叠为批次张量
                    batch_tensor = torch.stack(batch_images).to(self.device)
                    
                    with torch.no_grad():
                        embeddings = self.image_model(batch_tensor)
                        
                        # 如果输出是多维的，进行全局平均池化
                        if embeddings.dim() > 2:
                            embeddings = torch.nn.functional.adaptive_avg_pool2d(embeddings, (1, 1))
                            embeddings = embeddings.view(embeddings.size(0), -1)
                        
                        # 转为CPU numpy数组
                        embeddings = embeddings.cpu().numpy()
                        
                        # 保存embeddings
                        for j, valid_idx in enumerate(valid_indices):
                            img_id = batch_ids[valid_idx]
                            embeddings_dict[str(img_id)] = embeddings[j]
                            
                except Exception as e:
                    logger.error(f"处理图像批次 {i//self.batch_size + 1} 时出错: {e}")
                    # 逐张处理失败的批次
                    for j, valid_idx in enumerate(valid_indices):
                        try:
                            img_tensor = batch_images[j].unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                embedding = self.image_model(img_tensor)
                                if embedding.dim() > 2:
                                    embedding = torch.nn.functional.adaptive_avg_pool2d(embedding, (1, 1))
                                    embedding = embedding.view(1, -1)
                                embedding = embedding.cpu().numpy()[0]
                                
                            img_id = batch_ids[valid_idx]
                            embeddings_dict[str(img_id)] = embedding
                            
                        except Exception as e2:
                            logger.error(f"处理单张图像 {batch_ids[valid_idx]} 失败: {e2}")
                            # 使用零向量作为fallback
                            embeddings_dict[str(batch_ids[valid_idx])] = np.zeros(image_embedding_dim)  # DINOv2默认维度
        
        logger.info(f"图像embedding提取完成，成功提取 {len(embeddings_dict)} 张")
        return embeddings_dict

def generate_training_data_files():
    """生成训练代码需要的数据文件"""
    logger.info("\n" + "="*70)
    logger.info("开始生成训练所需的数据文件...")
    logger.info("="*70)
    
    # 1. 生成 tweet_embeddings.json
    logger.info("\n[1/3] 生成 tweet_embeddings.json...")
    
    # 加载已生成的embeddings
    with open('../../../autodl-tmp/pheme/original_text_embeddings.json', 'r') as f:
        original_emb = json.load(f)
    
    with open('../../../autodl-tmp/pheme/reply_embeddings.json', 'r') as f:
        reply_data = json.load(f)
        reply_emb = reply_data['embeddings']
    
    # 组合成训练代码需要的格式
    tweet_embeddings = {
        'original_posts': original_emb,
        'reply_posts': reply_emb
    }
    
    with open('../../../autodl-tmp/pheme/tweet_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(tweet_embeddings, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ tweet_embeddings.json 生成成功")
    logger.info(f"  - 原始帖子: {len(original_emb)} 条")
    logger.info(f"  - 回复帖子: {len(reply_emb)} 条")
    
    # 2. 生成 post_tweet.csv
    logger.info("\n[2/3] 生成 post_tweet.csv...")
    
    content_df = pd.read_csv('../content.csv')
    post_df = pd.DataFrame({
        'id_str': content_df['mid'].astype(str),
        'is_rumor': content_df['label'].astype(int)
    })
    post_df.to_csv('../../../autodl-tmp/pheme/post_tweet.csv', index=False)
    
    logger.info(f"✓ post_tweet.csv 生成成功: {len(post_df)} 条记录")
    
    # 3. 生成 comment_tweet.csv
    logger.info("\n[3/3] 生成 comment_tweet.csv...")
    
    with open('extracted_tweets.json', 'r', encoding='utf-8') as f:
        tweets_data = json.load(f)
    
    comment_records = []
    
    for original_id, replies in tweets_data.get('reaction_tweets', {}).items():
        for reply in replies:
            if 'data' in reply:
                reply_id = reply['data'].get('id_str', reply['filename'].replace('.json', ''))
                
                # 获取 in_reply_to_status_id_str
                in_reply_to = reply['data'].get('in_reply_to_status_id_str', original_id)
                
                comment_records.append({
                    'id_str': str(reply_id),
                    'in_reply_to_status_id_str': str(in_reply_to)
                })
    
    comment_df = pd.DataFrame(comment_records)
    comment_df.to_csv('../../../autodl-tmp/pheme/comment_tweet.csv', index=False)
    
    logger.info(f"✓ comment_tweet.csv 生成成功: {len(comment_df)} 条记录")
    
    # 4. 生成数据完整性报告
    logger.info("\n" + "="*70)
    logger.info("所有训练数据文件生成完成!")
    logger.info("="*70)
    logger.info("\n生成的文件清单:")
    logger.info(f"  ✓ tweet_embeddings.json - 传播链embeddings ({len(original_emb)} 原始 + {len(reply_emb)} 回复)")
    logger.info(f"  ✓ post_tweet.csv - 原始帖子数据 ({len(post_df)} 条)")
    logger.info(f"  ✓ comment_tweet.csv - 评论数据 ({len(comment_df)} 条)")
    logger.info(f"  ✓ content.csv - 文本图像内容 (已存在)")
    logger.info(f"  ✓ extracted_images/ - 图像文件 (已存在)")
    logger.info("\n可以开始运行训练代码了!")
    logger.info("="*70)

def extract_pheme_embeddings():
    """
    提取PHEME数据集的所有embeddings
    """
    # 初始化提取器 - 只使用本地Qwen3模型
    extractor = EmbeddingExtractor(
        qwen_model_path='../../../autodl-tmp/qwen3',  # 本地Qwen3-Embedding-8B模型路径
        device='cuda',
        batch_size=4  # 根据显存调整
    )
    
    # 1. 加载content.csv，提取原始推文信息
    logger.info("读取content.csv...")
    content_df = pd.read_csv('../content.csv')
    
    # 原始推文文本和图像
    original_texts = content_df['text'].astype(str).tolist()
    original_ids = content_df['mid'].astype(str).tolist()
    image_ids = content_df['imgnum'].astype(str).tolist()
    
    # 构建图像路径
    image_paths = []
    for img_id in image_ids:
        img_path = os.path.join('extracted_images', f'{img_id}.jpg')
        image_paths.append(img_path)
    
    logger.info(f"找到 {len(original_texts)} 条原始推文和 {len(image_paths)} 张图像")
    
    # 2. 加载extracted_tweets.json，提取回复推文信息
    logger.info("读取extracted_tweets.json...")
    with open('extracted_tweets.json', 'r', encoding='utf-8') as f:
        tweets_data = json.load(f)
    
    # 收集所有回复推文
    reply_texts = []
    reply_ids = []
    reply_to_original = {}  # 回复ID -> 原始推文ID的映射
    
    for original_id, replies in tweets_data.get('reaction_tweets', {}).items():
        for reply in replies:
            if 'data' in reply and 'text' in reply['data']:
                reply_text = reply['data']['text']
                reply_id = reply['data'].get('id_str', reply['filename'].replace('.json', ''))
                
                reply_texts.append(str(reply_text))
                reply_ids.append(str(reply_id))
                reply_to_original[str(reply_id)] = str(original_id)
    
    logger.info(f"找到 {len(reply_texts)} 条回复推文")
    
    # 3. 提取embeddings
    logger.info("=" * 50)
    logger.info("开始提取embeddings...")
    
    # 提取原始推文文本embeddings
    logger.info("提取原始推文文本embeddings...")
    original_task = 'Given a social media post about a news event, generate embedding for rumor detection and content analysis'
    original_text_embeddings = extractor.extract_text_embeddings(
        original_texts, 
        original_ids,
        custom_task=original_task
    )
    
    # 提取图像embeddings
    logger.info("提取图像embeddings...")
    image_embeddings = extractor.extract_image_embeddings(image_paths, image_ids)
    
    # 提取回复推文embeddings
    logger.info("提取回复推文embeddings...")
    reply_task = 'Given a social media reply post, generate embedding for sentiment analysis and response categorization'
    reply_embeddings = extractor.extract_text_embeddings(
        reply_texts, 
        reply_ids,
        custom_task=reply_task
    )
    
    # 4. 保存embeddings
    logger.info("保存embeddings到文件...")
    
    # 保存原始推文文本embeddings
    with open('../../../autodl-tmp/pheme/original_text_embeddings.pkl', 'wb') as f:
        pickle.dump(original_text_embeddings, f)
    
    # 同时保存json格式（便于查看，但文件较大）
    original_text_json = {k: v.tolist() for k, v in original_text_embeddings.items()}
    with open('../../../autodl-tmp/pheme/original_text_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(original_text_json, f, ensure_ascii=False, indent=2)
    
    # 保存图像embeddings
    with open('../../../autodl-tmp/pheme/image_embeddings.pkl', 'wb') as f:
        pickle.dump(image_embeddings, f)
    
    image_json = {k: v.tolist() for k, v in image_embeddings.items()}
    with open('../../../autodl-tmp/pheme/image_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(image_json, f, ensure_ascii=False, indent=2)
    
    # 保存回复推文embeddings（包含原始推文映射）
    reply_data = {
        'embeddings': reply_embeddings,
        'reply_to_original_mapping': reply_to_original
    }
    with open('../../../autodl-tmp/pheme/reply_embeddings.pkl', 'wb') as f:
        pickle.dump(reply_data, f)
    
    reply_json = {
        'embeddings': {k: v.tolist() for k, v in reply_embeddings.items()},
        'reply_to_original_mapping': reply_to_original
    }
    with open('../../../autodl-tmp/pheme/reply_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(reply_json, f, ensure_ascii=False, indent=2)
    
    # 5. 生成统计报告
    logger.info("=" * 50)
    logger.info("Embedding提取完成!")
    logger.info(f"原始推文文本embeddings: {len(original_text_embeddings)} 条")
    logger.info(f"图像embeddings: {len(image_embeddings)} 张")
    logger.info(f"回复推文embeddings: {len(reply_embeddings)} 条")
    
    # 检查embedding维度
    text_dim = 'unknown'
    if original_text_embeddings:
        text_dim = next(iter(original_text_embeddings.values())).shape[0]
        logger.info(f"文本embedding维度: {text_dim}")
    
    img_dim = 'unknown'
    if image_embeddings:
        img_dim = next(iter(image_embeddings.values())).shape[0]
        logger.info(f"图像embedding维度: {img_dim}")
    
    # 保存配置信息
    config = {
        'text_model': 'Qwen3-Embedding-8B',
        'image_model': extractor.image_model_name,
        'text_embedding_dim': int(text_dim) if isinstance(text_dim, (int, np.integer)) else text_dim,
        'image_embedding_dim': int(img_dim) if isinstance(img_dim, (int, np.integer)) else img_dim,
        'total_original_posts': len(original_text_embeddings),
        'total_images': len(image_embeddings),
        'total_replies': len(reply_embeddings),
        'extraction_date': pd.Timestamp.now().isoformat(),
        'original_task_instruction': original_task,
        'reply_task_instruction': reply_task
    }
    
    with open('embedding_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info("所有Embedding文件已保存!")
    logger.info("生成的文件:")
    logger.info("- original_text_embeddings.pkl/json: 原始推文文本embeddings")
    logger.info("- image_embeddings.pkl/json: 图像embeddings") 
    logger.info("- reply_embeddings.pkl/json: 回复推文embeddings")
    logger.info("- embedding_config.json: 配置信息")

if __name__ == "__main__":
    try:
        # 步骤1: 提取所有embeddings
        extract_pheme_embeddings()
        
        # 步骤2: 生成训练所需的数据文件
        generate_training_data_files()
        
        logger.info("\n" + "="*70)
        logger.info("数据预处理完成! 可以开始训练模型了")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
