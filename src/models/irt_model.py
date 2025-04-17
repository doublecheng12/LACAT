"""
IRT (Item Response Theory) 模型实现。
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
        logging.FileHandler("logs/irt_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IRTNet(nn.Module):
    """IRT神经网络模型"""
    
    def __init__(self, student_num: int, question_num: int):
        """
        初始化IRT网络
        
        Args:
            student_num: 学生数量
            question_num: 问题数量
        """
        super(IRTNet, self).__init__()
        self.student_num = student_num
        self.question_num = question_num
        
        # 初始化参数
        self.theta = nn.Embedding(student_num, 1)
        self.alpha = nn.Embedding(question_num, 1)
        self.beta = nn.Embedding(question_num, 1)
        
        # 初始化权重
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, student_ids: torch.Tensor, question_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            student_ids: 学生ID张量
            question_ids: 问题ID张量
            
        Returns:
            预测概率
        """
        theta = self.theta(student_ids)
        alpha = self.alpha(question_ids)
        beta = self.beta(question_ids)
        
        # IRT模型公式: P(theta) = c + (1-c) / (1 + exp(-alpha*(theta-beta)))
        # 这里简化为 P(theta) = sigmoid(alpha*(theta-beta))
        logits = alpha * (theta - beta)
        return torch.sigmoid(logits)


class IRTModel:
    """IRT模型包装器，提供训练和评估功能"""
    
    def __init__(
        self, 
        lr: float = 0.003, 
        batch_size: int = 32, 
        epochs: int = 30, 
        device: Optional[str] = None
    ):
        """
        初始化IRT模型
        
        Args:
            lr: 学习率
            batch_size: 批量大小
            epochs: 训练轮次
            device: 计算设备
        """
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.BCELoss()
        
        logger.info(f"初始化IRT模型，设备: {self.device}")

    def init_model(self, data: Any) -> None:
        """
        初始化模型
        
        Args:
            data: 包含学生和问题信息的数据对象
        """
        # 获取学生和问题数量
        student_num = data.num_students
        question_num = data.num_questions
        
        # 创建模型
        self.model = IRTNet(student_num, question_num).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        logger.info(f"模型初始化完成，学生数: {student_num}，问题数: {question_num}")

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
            pred = self.model(student_ids, question_ids).squeeze()
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
                pred = self.model(student_ids, question_ids).squeeze()
                
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

    def get_theta(self, student_id: int) -> np.ndarray:
        """
        获取学生的能力参数
        
        Args:
            student_id: 学生ID
            
        Returns:
            能力参数
        """
        with torch.no_grad():
            theta = self.model.theta.weight[student_id].cpu().numpy()
        return theta

    def get_question_params(self, question_id: int) -> Tuple[float, float]:
        """
        获取问题的区分度和难度参数
        
        Args:
            question_id: 问题ID
            
        Returns:
            区分度和难度参数元组
        """
        with torch.no_grad():
            alpha = self.model.alpha.weight[question_id].item()
            beta = self.model.beta.weight[question_id].item()
        return alpha, beta

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
        pred = self.model(student_ids, question_ids).squeeze()
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