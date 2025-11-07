import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torchvision.models import resnet34, ResNet34_Weights


class PermuteLayer(nn.Module):
    """维度置换辅助层"""

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding='max_length', truncation=True, max_length=32
        ).to(next(self.parameters()).device)
        outputs = self.bert(**inputs)
        return outputs.last_hidden_state  # [B, S, D]


class VideoEncoder(nn.Module):
    def __init__(self, embed_dim=768, target_frames=32):
        super().__init__()
        self.target_frames = target_frames

        # 修正后的时空特征编码器
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(4096, embed_dim, 3, padding=1),  # [B, 4096, T] -> [B, D, T]
            nn.ReLU(),
            PermuteLayer(0, 2, 1),  # [B, D, T] -> [B, T, D]
            nn.LayerNorm(embed_dim),  # 在特征维度归一化
        )

        # 运动建模GRU
        self.motion_gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim // 2,
            bidirectional=True,
            batch_first=True
        )

        # 动态时间对齐
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_frames)

    def forward(self, x):
        """Input: [B, T, 4096]"""
        # 维度调整
        x = x.permute(0, 2, 1)  # [B, 4096, T]

        # 自适应池化统一时间维度
        x = self.adaptive_pool(x)  # [B, 4096, target_T]

        # 时空特征提取
        base_feat = self.temporal_conv(x)  # [B, target_T, D]

        # 运动特征提取
        motion_feat, _ = self.motion_gru(base_feat)  # [B, T, D]

        return motion_feat  # [B, T, D]


class CrossModalAligner(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        # 双向跨模态注意力
        self.video_attn = nn.MultiheadAttention(embed_dim, 4, batch_first=True)
        self.text_attn = nn.MultiheadAttention(embed_dim, 4, batch_first=True)

        # 时间-语义聚合
        self.time_weight = nn.Parameter(torch.ones(1, 32, 1))  # [1, T, 1]
        self.cls_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, video_feat, text_feat):
        """
        video_feat: [B, T, D]
        text_feat: [B, S, D]
        """
        # 视频→文本注意力
        video_attended, _ = self.video_attn(
            query=video_feat,
            key=text_feat,
            value=text_feat
        )

        # 文本→视频注意力
        text_attended, _ = self.text_attn(
            query=text_feat,
            key=video_feat,
            value=video_feat
        )

        # 全局特征聚合
        video_global = torch.matmul(
            F.softmax(self.time_weight, dim=1).permute(0, 2, 1),  # [1, 1, T]
            video_attended  # [B, T, D]
        ).squeeze(1)  # [B, D]

        text_global = self.cls_proj(text_attended[:, 0])  # CLS token

        # 特征归一化
        video_global = F.normalize(video_global, p=2, dim=-1)
        text_global = F.normalize(text_global, p=2, dim=-1)

        return video_global, text_global, video_attended, text_attended


class HierarchicalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor([1.0]).log())

    def compute_local_similarity(self, video_local, text_local, logit_scale):
        """计算局部相似度矩阵"""
        batch_size, T, D = video_local.shape
        S = text_local.size(1)
        
        # 计算帧-token相似度矩阵 [B, T, S]
        local_sim = torch.einsum('btd,bsd->bts', video_local, text_local)
        local_sim = logit_scale * local_sim
        
        return local_sim

    def compute_positive_scores(self, local_sim):
        """计算正样本分数 - 与正文公式一致"""
        batch_size, T, S = local_sim.shape
        
        # 时间敏感权重计算 [B, T]
        time_weights = F.softmax(local_sim.max(dim=-1).values, dim=-1)
        
        # 正样本分数计算: 每帧与所有token的平均相似度 [B, T]
        frame_text_similarity = local_sim.mean(dim=-1)  # [B, T]
        
        # 加权聚合 [B]
        pos_scores = (frame_text_similarity * time_weights).sum(dim=1)
        
        return pos_scores, time_weights

    def compute_negative_scores(self, local_sim_pairs, time_weights):
        """计算负样本分数 - 与正文公式一致"""
        batch_size = local_sim_pairs.shape[0]
        
        # 计算所有负样本对的相似度矩阵
        neg_scores = []
        for i in range(batch_size):
            sample_neg_scores = []
            for j in range(batch_size):
                if i == j:
                    continue  # 跳过正样本对
                
                # 计算样本i与样本j的局部相似度矩阵 [T, S]
                local_sim_ij = local_sim_pairs[i, j]  # 假设已经预计算好
                
                # 使用样本i的时间权重加权平均 [T] -> scalar
                neg_score_ij = (time_weights[i] * local_sim_ij.mean(dim=-1)).sum()
                sample_neg_scores.append(neg_score_ij)
            
            neg_scores.append(torch.stack(sample_neg_scores))
        
        neg_scores = torch.stack(neg_scores)  # [B, B-1]
        return neg_scores

    def forward(self, video_global, text_global, video_local, text_local):
        logit_scale = self.temp.exp()
        
        # 全局对比损失
        global_sim = logit_scale * (video_global @ text_global.t())
        loss_global = (F.cross_entropy(global_sim, torch.arange(len(video_global), device=video_global.device)) +
                      F.cross_entropy(global_sim.t(), torch.arange(len(text_global), device=text_global.device))) / 2

        # 局部对比损失
        batch_size = video_local.size(0)
        
        # 计算正样本对的局部相似度
        local_sim_positive = self.compute_local_similarity(video_local, text_local, logit_scale)
        
        # 计算正样本分数
        pos_scores, time_weights = self.compute_positive_scores(local_sim_positive)
        
        # 计算负样本分数 (简化实现 - 在实际应用中需要预计算所有样本对的相似度)
        # 这里使用当前批次内其他样本作为负样本
        neg_scores = []
        for i in range(batch_size):
            sample_neg_scores = []
            for j in range(batch_size):
                if i == j:
                    continue
                # 使用样本i的视频特征和样本j的文本特征计算负样本相似度
                local_sim_neg = self.compute_local_similarity(
                    video_local[i:i+1], text_local[j:j+1], logit_scale
                ).squeeze(0)  # [T, S]
                
                # 加权平均计算负样本分数
                neg_score = (time_weights[i] * local_sim_neg.mean(dim=-1)).sum()
                sample_neg_scores.append(neg_score)
            
            neg_scores.append(torch.stack(sample_neg_scores))
        
        neg_scores = torch.stack(neg_scores)  # [B, B-1]
        
        # 组合损失
        logits = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)  # [B, 1 + (B-1)]
        loss_local = F.cross_entropy(logits, torch.zeros(batch_size, dtype=torch.long, device=video_global.device))

        return 0.6 * loss_global + 0.4 * loss_local


class VideoTextModel(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder(embed_dim)
        self.image_encoder = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.image_encoder.fc = nn.Linear(512, embed_dim)

        self.aligner = CrossModalAligner(embed_dim)
        self.loss_fn = HierarchicalLoss()
        self.head1 = nn.Sequential(
            nn.Linear(embed_dim, 256), 
            nn.Dropout(0.5), 
            nn.ReLU(), 
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.head2 = nn.Sequential(
            nn.Linear(embed_dim, 256), 
            nn.Dropout(0.5), 
            nn.ReLU(), 
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, images, videos, texts):
        # 特征提取
        img_feat = self.image_encoder(images)
        text_feat = self.text_encoder(texts)  # [B, S, D]
        video_feat = self.video_encoder(videos)  # [B, T, D]

        # 跨模态对齐
        v_global, t_global, v_local, t_local = self.aligner(video_feat, text_feat)

        # 多粒度损失
        align_loss = self.loss_fn(v_global, t_global, v_local, t_local)

        # 特征融合
        score1 = self.head1(img_feat).squeeze()
        score2 = self.head2(v_global).squeeze()

        return score1, score2, align_loss


# 示例用法
if __name__ == "__main__":
    # 初始化模型
    model = VideoTextModel().cuda()

    # 创建测试数据
    dummy_images = torch.randn(4, 3, 224, 224).cuda()
    dummy_videos = torch.randn(4, 30, 4096).cuda()  # 原始视频特征 (B, T, 4096)
    dummy_texts = ["a person dancing", "a dog running",
                   "cars moving fast", "water flowing"]

    # 前向传播测试
    score1, score2, loss = model(dummy_images, dummy_videos, dummy_texts)
    print(f"Score1 shape: {score1.shape}")  # 应输出 torch.Size([4])
    print(f"Score2 shape: {score2.shape}")  # 应输出 torch.Size([4])
    print(f"Loss value: {loss.item():.4f}")  # 合理损失值
