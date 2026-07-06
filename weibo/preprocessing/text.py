import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import json

class TextEncoder:
    def __init__(self, model_name='hfl/chinese-roberta-wwm-ext', batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.batch_size = batch_size
        self.max_length = 128  # 根据实际文本长度调整

    def encode_batch(self, texts):
        # 对一批文本进行编码
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]标记的输出作为文本表示
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings

    def encode_all(self, texts):
        all_embeddings = []
        # 使用tqdm显示进度
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            embeddings = self.encode_batch(batch_texts)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

def process_and_save_embeddings():
    # 读取CSV文件
    print("Reading CSV files...")
    posts_df = pd.read_csv('weibo_ori.csv')
    comments_df = pd.read_csv('weibo_reply.csv')
    
    # 打印原始数据信息
    print("\n=== 原始数据信息 ===")
    print("原始推文总数:", len(posts_df))
    print("回复推文总数:", len(comments_df))
    
    # 初始化编码器
    print("\nInitializing RoBERTa encoder...")
    encoder = TextEncoder()
    
    # 处理原始推文
    print("Encoding original posts...")
    post_embeddings = encoder.encode_all(posts_df['text'].tolist())
    
    # 处理评论推文
    print("Encoding reply posts...")
    comment_embeddings = encoder.encode_all(comments_df['text'].tolist())
    
    # 创建包含id和embeddings的字典
    embeddings_dict = {
        'original_posts': {
            str(id_str): emb.tolist() 
            for id_str, emb in zip(posts_df['id_str'], post_embeddings)
        },
        'reply_posts': {
            str(mid): emb.tolist() 
            for mid, emb in zip(comments_df['mid'], comment_embeddings)
        }
    }
    
    # 保存结果
    print("\nSaving embeddings...")
    with open('weibo_embeddings.json', 'w') as f:
        json.dump(embeddings_dict, f)
    
    # 保存一些基本统计信息
    stats = {
        'num_original_posts': len(posts_df),
        'num_reply_posts': len(comments_df),
        'embedding_dim': post_embeddings.shape[1],
        'original_posts_ids': posts_df['id_str'].tolist(),
        'reply_posts_ids': comments_df['mid'].tolist()
    }
    
    with open('embedding_stats.json', 'w') as f:
        json.dump(stats, f)
    
    print("\n=== 最终统计 ===")
    print(f"处理完成的原始推文数量: {len(embeddings_dict['original_posts'])}")
    print(f"处理完成的回复推文数量: {len(embeddings_dict['reply_posts'])}")
    print("结果已保存到 'weibo_embeddings.json'")

if __name__ == "__main__":
    process_and_save_embeddings()
