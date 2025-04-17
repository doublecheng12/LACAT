"""
NCD (Neural Cognitive Diagnosis) 模型实现。
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, accuracy_score

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ncd_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NCDNet(nn.Module):
    """NCD神经网络模型"""
    
    def __init__(
        self, 
        student_num: int, 
        question_num: int, 
        knowledge_num: int,
        knowledge_dim: int = 16
    ):
        """
        初始化NCD网络
        
        Args:
            student_num: 学生数量
            question_num: 问题数量
            knowledge_num: 知识点数量
            knowledge_dim: 知识点维度
        """
        super(NCDNet, self).__init__()
        self.student_num = student_num
        self.question_num = question_num
        self.knowledge_num = knowledge_num
        self.knowledge_dim = knowledge_dim
        
        # 学生知识熟练度矩阵
        self.student_emb = nn.Embedding(student_num, knowledge_num)
        
        # 问题-知识点关联矩阵
        self.question_discriminative = nn.Embedding(question_num, knowledge_num)
        self.question_difficulty = nn.Embedding(question_num, 1)
        
        # 知识点嵌入
        self.knowledge_emb = nn.Embedding(knowledge_num, knowledge_dim)
        
        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(knowledge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(
        self, 
        student_ids: torch.Tensor, 
        question_ids: torch.Tensor, 
        knowledge_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            student_ids: 学生ID张量
            question_ids: 问题ID张量
            knowledge_matrix: 知识点关联矩阵（可选）
            
        Returns:
            预测概率
        """
        # 获取学生知识熟练度
        stu_knowledge = torch.sigmoid(self.student_emb(student_ids))  # [batch_size, knowledge_num]
        
        # 获取问题的知识点要求和难度
        q_knowledge = torch.sigmoid(self.question_discriminative(question_ids))  # [batch_size, knowledge_num]
        q_difficulty = torch.sigmoid(self.question_difficulty(question_ids))  # [batch_size, 1]
        
        # 计算学生知识熟练度和问题知识点要求的差异
        diff = stu_knowledge - q_knowledge
        
        # 使用MLP处理知识点差异
        knowledge_features = torch.matmul(diff, self.knowledge_emb.weight)  # [batch_size, knowledge_dim]
        pred = self.mlp(knowledge_features)  # [batch_size, 1]
        
        # 应用难度修正
        pred = pred * (1 - q_difficulty)
        
        return pred.squeeze()

    def get_knowledge_status(self, student_id: int) -> np.ndarray:
        """
        获取学生的知识点掌握状态
        
        Args:
            student_id: 学生ID
            
        Returns:
            知识点掌握状态
        """
        with torch.no_grad():
            status = torch.sigmoid(self.student_emb.weight[student_id]).cpu().numpy()
        return status


class NCDModel:
    """NCD模型包装器，提供训练和评估功能"""
    
    def __init__(
        self, 
        lr: float = 0.002, 
        batch_size: int = 32, 
        epochs: int = 30,
        knowledge_dim: int = 16,
        device: Optional[str] = None
    ):
        """
        初始化NCD模型
        
        Args:
            lr: 学习率
            batch_size: 批量大小
            epochs: 训练轮次
            knowledge_dim: 知识点嵌入维度
            device: 计算设备
        """
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.knowledge_dim = knowledge_dim
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.BCELoss()
        
        logger.info(f"初始化NCD模型，设备: {self.device}")

    def init_model(self, data: Any) -> None:
        """
        初始化模型
        
        Args:
            data: 包含学生、问题和知识点信息的数据对象
        """
        # 获取学生、问题和知识点数量
        student_num = data.num_students
        question_num = data.num_questions
        knowledge_num = data.num_concepts
        
        # 创建模型
        self.model = NCDNet(
            student_num, 
            question_num, 
            knowledge_num,
            self.knowledge_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        logger.info(f"模型初始化完成，学生数: {student_num}，问题数: {question_num}，知识点数: {knowledge_num}")

    def train(self, data: Any) -> float:
        """
        训练模型
        
        Args:
            data: 训练数据
            
        Returns:
            训练损失
        """
        self.model.train()
        total_loss = 0.0
        
        # 创建数据加载器
        dataloader = data.get_dataloader(batch_size=self.batch_size, shuffle=True)
        
        for batch in dataloader:
            student_ids, question_ids, labels = [x.to(self.device) for x in batch]
            
            # 前向传播
            self.optimizer.zero_grad()
            pred = self.model(student_ids, question_ids)
            loss = self.loss_fn(pred, labels.float())
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        logger.info(f"训练完成，平均损失: {avg_loss:.4f}")
        
        return avg_loss

    def evaluate(self, data: Any) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            data: 评估数据
            
        Returns:
            包含评估指标的字典
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        # 创建数据加载器
        dataloader = data.get_dataloader(batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                student_ids, question_ids, labels = [x.to(self.device) for x in batch]
                
                # 前向传播
                pred = self.model(student_ids, question_ids)
                
                # 收集预测和标签
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 二值化预测
        binary_preds = (all_preds >= 0.5).astype(int)
        
        # 计算准确率和AUC
        acc = accuracy_score(all_labels, binary_preds)
        auc = roc_auc_score(all_labels, all_preds)
        
        logger.info(f"评估完成，准确率: {acc:.4f}，AUC: {auc:.4f}")
        
        return {'acc': acc, 'auc': auc}

    def get_knowledge_status(self, student_id: int) -> np.ndarray:
        """
        获取学生的知识掌握状态
        
        Args:
            student_id: 学生ID
            
        Returns:
            知识掌握状态
        """
        return self.model.get_knowledge_status(student_id)

    def adaptest_update(self, data: Any) -> float:
        """
        自适应测试更新模型
        
        Args:
            data: 更新数据
            
        Returns:
            更新损失
        """
        self.model.train()
        total_loss = 0.0
        
        # 获取最新数据
        student_ids, question_ids, labels = data.get_recent_data()
        student_ids = torch.tensor(student_ids).to(self.device)
        question_ids = torch.tensor(question_ids).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        
        # 前向传播
        self.optimizer.zero_grad()
        pred = self.model(student_ids, question_ids)
        loss = self.loss_fn(pred, labels.float())
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        total_loss += loss.item()
        
        logger.info(f"自适应更新完成，损失: {total_loss:.4f}")
        
        return total_loss

    def adaptest_load(self, model_path: str) -> None:
        """
        加载预训练模型
        
        Args:
            model_path: 模型路径
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            logger.info(f"成功加载模型: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}", exc_info=True)
            raise

    def adaptest_save(self, model_path: str) -> None:
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
        """
        try:
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"模型保存成功: {model_path}")
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}", exc_info=True)