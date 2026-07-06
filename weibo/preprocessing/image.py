import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import os
import pandas as pd
import open_clip
from torchvision import transforms
from PIL import Image
import numpy as np

# 设置路径
CSV_PATH = "../weibo/weibocontentwithimage/weibo_content.csv"
IMG_FOLDER = "../weibo/weibo_images/weibo_images_all"
TRAIN_ID_PATH = "../weibo/train_id.txt"
TEST_ID_PATH = "../weibo/test_id.txt"
DEV_ID_PATH = "../weibo/dev_id.txt"
SAVE_PATH = "weibo_features.npz"

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载CLIP模型
print("Loading CLIP model...")
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device)

# # 冻结CLIP底层权重，仅解冻高层权重
# for name, param in model.named_parameters():
#     if "ln_post" in name or "visual.proj" in name or "transformer.resblocks.11" in name:  # 解冻高层模块
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# 只解冻最后一层的权重
for name, param in model.named_parameters():
    if "transformer.resblocks.11" in name:  # 只解冻最后一层
        param.requires_grad = True
    else:
        param.requires_grad = False


# 查看可训练参数数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters in CLIP: {trainable_params}")

# 读取数据
print("Loading CSV data...")
data = pd.read_csv(CSV_PATH)

# 加载数据分割的id和标签
def load_ids(file_path):
    ids, labels = [], []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                ids.append(parts[0])
                labels.append(int(parts[1]))
    return ids, labels

train_ids, train_labels = load_ids(TRAIN_ID_PATH)
test_ids, test_labels = load_ids(TEST_ID_PATH)
dev_ids, dev_labels = load_ids(DEV_ID_PATH)

# 提取文本和图像特征
def extract_features(df, img_folder):
    text_features, image_features, labels = [], [], []
    for i, row in df.iterrows():
        imgnum = str(row["imgnum"]) + ".jpg"
        text = row["text"]
        label = row["label"]

        # 文本特征提取
        text_tokens = tokenizer([text]).to(device)
        text_feat = model.encode_text(text_tokens).float()

        # 图像特征提取
        img_path = os.path.join(img_folder, imgnum)
        if os.path.exists(img_path):
            image = preprocess_val(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            image_feat = model.encode_image(image).float()
        else:
            image_feat = torch.zeros((1, 512)).to(device)  # 图片不存在时，使用0向量

        text_features.append(text_feat.squeeze().detach().cpu().numpy())
        image_features.append(image_feat.squeeze().detach().cpu().numpy())
        labels.append(label)
    
    return np.array(text_features), np.array(image_features), np.array(labels)

# 划分数据集
train_df = data[data["mid"].isin(train_ids)]
test_df = data[data["mid"].isin(test_ids)]
dev_df = data[data["mid"].isin(dev_ids)]

print("Extracting train features...")
train_text, train_image, train_labels = extract_features(train_df, IMG_FOLDER)

print("Extracting test features...")
test_text, test_image, test_labels = extract_features(test_df, IMG_FOLDER)

print("Extracting dev features...")
dev_text, dev_image, dev_labels = extract_features(dev_df, IMG_FOLDER)


def contrastive_loss(text_features, image_features, temperature=0.07):
    """
    计算文本和图像特征之间的对比损失
    Args:
        text_features: 文本特征 (batch_size, feature_dim)
        image_features: 图像特征 (batch_size, feature_dim)
        temperature: 温度参数，控制softmax的分布
    Returns:
        对比损失值
    """
    # 归一化特征
    text_features = nn.functional.normalize(text_features, dim=1)
    image_features = nn.functional.normalize(image_features, dim=1)

    # 计算余弦相似度
    logits = torch.matmul(text_features, image_features.T) / temperature
    batch_size = logits.shape[0]

    # 生成标签，同一位置的文本和图像是正样本
    labels = torch.arange(batch_size).to(text_features.device)

    # 使用交叉熵损失作为对比损失
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss


# 融合文本和图像特征
def fuse_features(text_features, image_features):
    return np.concatenate([text_features, image_features], axis=1)

train_features = fuse_features(train_text, train_image)
test_features = fuse_features(test_text, test_image)
dev_features = fuse_features(dev_text, dev_image)

# 保存提取的特征和标签
print("Saving features...")
np.savez_compressed(
    SAVE_PATH,
    train_features=train_features,
    train_labels=train_labels,
    test_features=test_features,
    test_labels=test_labels,
    dev_features=dev_features,
    dev_labels=dev_labels,
)
print(f"Features saved to {SAVE_PATH}")

# 数据加载
data = np.load(SAVE_PATH)
train_features = torch.tensor(data["train_features"], dtype=torch.float32)
train_labels = torch.tensor(data["train_labels"], dtype=torch.long)
test_features = torch.tensor(data["test_features"], dtype=torch.float32)
test_labels = torch.tensor(data["test_labels"], dtype=torch.long)

# 定义数据加载器
batch_size = 64
train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=batch_size, shuffle=False)

# 定义深度学习模型
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 模型初始化
input_dim = train_features.shape[1]
hidden_dim = 512
num_classes = 2
classifier = Classifier(input_dim, hidden_dim, num_classes).to(device)

# 定义损失函数和优化器（包括CLIP参数和分类器参数）
optimizer = optim.Adam([
    {"params": model.parameters(), "lr": 1e-6},  # CLIP模型微调学习率较低
    {"params": classifier.parameters(), "lr": 1e-3}  # 分类器学习率较高
])
criterion = nn.CrossEntropyLoss()

# 训练模型
# 训练函数
def train(model, classifier, train_loader, criterion, optimizer, device, epochs=10, alpha=0.5):
    """
    训练函数
    Args:
        model: CLIP 模型
        classifier: 分类器
        train_loader: 数据加载器
        criterion: 分类损失函数 (交叉熵)
        optimizer: 优化器
        alpha: 对比损失的权重
    """
    model.train()
    classifier.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            # 将特征分解为文本和图像部分
            batch_size = batch_features.shape[0]
            text_features = batch_features[:, :512]  # 前512维是文本特征
            image_features = batch_features[:, 512:]  # 后512维是图像特征
            
            optimizer.zero_grad()

            # 分类器预测
            outputs = classifier(batch_features)
            loss_cls = criterion(outputs, batch_labels)

            # 计算对比损失
            loss_contrastive = contrastive_loss(
                torch.tensor(text_features).to(device), 
                torch.tensor(image_features).to(device)
            )

            # 总损失：分类损失 + α * 对比损失
            loss = loss_cls + alpha * loss_contrastive

            # 反向传播与优化
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")


# 评估模型
def evaluate(classifier, test_loader, device):
    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    return acc, precision, recall, f1

# 模型训练和评估
print("Training the model...")
train(model, classifier, train_loader, criterion, optimizer, device, epochs=100, alpha=0.5)

print("Evaluating the model on test set...")
acc, precision, recall, f1 = evaluate(classifier, test_loader, device)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")
