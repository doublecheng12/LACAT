"""
自适应测试数据集模块。
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/dataset.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CATDataset(Dataset):
    """自适应测试数据集类"""
    
    def __init__(
        self, 
        student_ids: List[int], 
        question_ids: List[int], 
        labels: List[int]
    ):
        """
        初始化数据集
        
        Args:
            student_ids: 学生ID列表
            question_ids: 问题ID列表
            labels: 标签列表
        """
        self.student_ids = torch.tensor(student_ids)
        self.question_ids = torch.tensor(question_ids)
        self.labels = torch.tensor(labels)
        
    def __len__(self) -> int:
        """
        获取数据集长度
        
        Returns:
            数据集长度
        """
        return len(self.student_ids)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取数据集项
        
        Args:
            idx: 索引
            
        Returns:
            学生ID、问题ID和标签元组
        """
        return self.student_ids[idx], self.question_ids[idx], self.labels[idx]


class AdapTestDataset:
    """自适应测试数据集包装器，提供数据加载和处理功能"""
    
    def __init__(
        self, 
        data_dir: str,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True
    ):
        """
        初始化数据集包装器
        
        Args:
            data_dir: 数据目录
            batch_size: 批量大小
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            shuffle: 是否打乱数据
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        
        # 加载数据
        self._load_data()
        
        # 分割数据
        self._split_data()
        
        # 最近选择记录
        self.recent_selections = []
        
        logger.info(f"数据集初始化完成，学生数: {self.num_students}，问题数: {self.num_questions}")
    
    def _load_data(self) -> None:
        """加载数据"""
        try:
            # 加载三元组数据（学生、问题、正确性）
            triples_path = os.path.join(self.data_dir, "train_triples.csv")
            self.triples_df = pd.read_csv(triples_path)
            
            # 加载概念映射
            concept_map_path = os.path.join(self.data_dir, "concept_map.json")
            if os.path.exists(concept_map_path):
                import json
                with open(concept_map_path, 'r', encoding='utf-8') as f:
                    self.concept_map = json.load(f)
            else:
                self.concept_map = {}
                
            # 构建学生-问题-结果字典
            self.data = {}
            for _, row in self.triples_df.iterrows():
                sid = int(row['student_id'])
                qid = int(row['question_id'])
                correct = int(row['correct'])
                
                if sid not in self.data:
                    self.data[sid] = {}
                    
                self.data[sid][qid] = correct
                
            # 获取学生和问题数量
            self.num_students = self.triples_df['student_id'].nunique()
            self.num_questions = self.triples_df['question_id'].nunique()
            
            # 获取概念数量
            self.num_concepts = 0
            if self.concept_map:
                # 找出所有概念的ID
                all_concepts = set()
                for qid, concepts in self.concept_map.items():
                    all_concepts.update(concepts)
                self.num_concepts = len(all_concepts)
            
            logger.info(f"数据加载成功，学生数: {self.num_students}，问题数: {self.num_questions}，概念数: {self.num_concepts}")
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}", exc_info=True)
            raise
            
    def _split_data(self) -> None:
        """分割数据集"""
        # 获取所有唯一的学生ID
        student_ids = sorted(list(self.data.keys()))
        np.random.shuffle(student_ids)
        
        # 计算分割点
        train_end = int(len(student_ids) * self.train_ratio)
        val_end = train_end + int(len(student_ids) * self.val_ratio)
        
        # 分割数据
        self.train_students = student_ids[:train_end]
        self.val_students = student_ids[train_end:val_end]
        self.test_students = student_ids[val_end:]
        
        logger.info(f"数据分割完成，训练集: {len(self.train_students)}，验证集: {len(self.val_students)}，测试集: {len(self.test_students)}")
    
    def get_dataloader(
        self, 
        batch_size: Optional[int] = None, 
        shuffle: Optional[bool] = None,
        split: str = 'train'
    ) -> DataLoader:
        """
        获取数据加载器
        
        Args:
            batch_size: 批量大小（可选）
            shuffle: 是否打乱数据（可选）
            split: 数据集分割（'train', 'val', 'test'）
            
        Returns:
            数据加载器
        """
        # 设置默认值
        batch_size = batch_size or self.batch_size
        shuffle = shuffle if shuffle is not None else self.shuffle
        
        # 选择学生集
        if split == 'train':
            students = self.train_students
        elif split == 'val':
            students = self.val_students
        elif split == 'test':
            students = self.test_students
        else:
            raise ValueError(f"未知的数据集分割: {split}")
        
        # 收集三元组
        student_ids = []
        question_ids = []
        labels = []
        
        for sid in students:
            for qid, correct in self.data[sid].items():
                student_ids.append(sid)
                question_ids.append(qid)
                labels.append(correct)
        
        # 创建数据集和数据加载器
        dataset = CATDataset(student_ids, question_ids, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return dataloader
    
    def apply_selection(self, student_id: int, question_id: int) -> None:
        """
        应用问题选择
        
        Args:
            student_id: 学生ID
            question_id: 问题ID
        """
        # 检查是否存在
        if student_id in self.data and question_id in self.data[student_id]:
            # 记录选择
            correct = self.data[student_id][question_id]
            self.recent_selections.append((student_id, question_id, correct))
            logger.debug(f"应用选择: 学生 {student_id}, 问题 {question_id}, 结果 {correct}")
        else:
            logger.warning(f"找不到学生 {student_id} 的问题 {question_id} 记录")
    
    def get_recent_data(self) -> Tuple[List[int], List[int], List[int]]:
        """
        获取最近的选择数据
        
        Returns:
            学生ID、问题ID和标签列表的元组
        """
        student_ids = []
        question_ids = []
        labels = []
        
        for sid, qid, correct in self.recent_selections:
            student_ids.append(sid)
            question_ids.append(qid)
            labels.append(correct)
        
        self.recent_selections = []  # 清空记录
        
        return student_ids, question_ids, labels