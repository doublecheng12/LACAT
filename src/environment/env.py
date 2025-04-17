"""
环境模块，实现计算机自适应测试环境。
"""

import logging
import numpy as np
import random
import torch
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from src.utils.data_utils import get_problem_detail, get_embedding

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/environment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Environment:
    """自适应测试环境"""
    
    def __init__(self, data: Any, concept_map: Dict, config: Dict, T: int):
        """
        初始化环境
        
        Args:
            data: 测试数据
            concept_map: 概念映射
            config: 配置参数
            T: 测试长度
        """
        self.last_ACC = None
        self.last_rate = None
        self.last_AUC = None
        self.config = config
        self.T = T
        self.CDM = 'IRT'  # 默认使用IRT模型
        self.rates = {}
        self.users = {}
        self.utypes = {}
        self.dataset_name = 'moocradar'  # 默认数据集
        self.tested_embs = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 加载数据
        self.rates, self._item_num, self.know_map = self.load_data(data, concept_map)
        self.tsdata = data
        
        # 设置训练和测试集
        self.setup_train_test()
        
        # 初始化记录
        self.records = []
        self.knowledge = []
        
        # 分割数据
        self.sup_rates, self.query_rates = self.split_data(ratio=0.8)
        
        # 加载模型
        pth_path = 'ckpt/moocradar/irt.pt'
        self.selected = []
        self.model, self.dataset = self.load_CDM('IRT', data, pth_path, config)
        
        logger.info(f"环境初始化完成, 数据集: {self.dataset_name}, 项目数: {self._item_num}")

    def split_data(self, ratio: float = 1.0) -> Tuple[Dict, Dict]:
        """
        分割数据为支持集和查询集
        
        Args:
            ratio: 分割比例
            
        Returns:
            支持集和查询集
        """
        sup_rates, query_rates = {}, {}
        
        for u in self.rates:
            all_items = list(self.rates[u].keys())
            np.random.shuffle(all_items)
            split_idx = int(ratio * len(all_items))
            
            sup_rates[u] = {it: self.rates[u][it] for it in all_items[:split_idx]}
            query_rates[u] = {it: self.rates[u][it] for it in all_items[split_idx:]}
            
        return sup_rates, query_rates

    def re_split_data(self, ratio: float = 0.5) -> None:
        """
        重新分割数据
        
        Args:
            ratio: 分割比例
        """
        self.sup_rates, self.query_rates = self.split_data(ratio)

    @property
    def candidate_items(self) -> set:
        """
        获取候选项目
        
        Returns:
            候选项目集合
        """
        return set(self.sup_rates[self.state[0][0]].keys())

    @property
    def user_num(self) -> int:
        """
        获取用户数量
        
        Returns:
            用户数量
        """
        return len(self.rates) + 1

    @property
    def item_num(self) -> int:
        """
        获取项目数量
        
        Returns:
            项目数量
        """
        return self._item_num + 1

    @property
    def utype_num(self) -> int:
        """
        获取用户类型数量
        
        Returns:
            用户类型数量
        """
        return len(self.utypes) + 1

    def setup_train_test(self) -> None:
        """设置训练和测试集"""
        users = list(range(1, self.user_num))
        np.random.shuffle(users)
        
        self.training, self.validation, self.evaluation = np.split(
            np.asarray(users), 
            [int(.8 * self.user_num - 1), int(.9 * self.user_num - 1)]
        )

    def reset(self) -> None:
        """重置环境"""
        self.reset_with_users(np.random.choice(self.training))

    def reset_with_users(self, uid: int, action: int, infos: Dict) -> Tuple:
        """
        使用特定用户重置环境
        
        Args:
            uid: 用户ID
            action: 动作ID
            infos: 信息集
            
        Returns:
            theta, selected, observation
        """
        self.state = [(uid, 1), []]
        self.short = {}
        
        # 执行步骤
        self.state, _, self.observation, self.records, done, self.selected, self.theta = self.step(action, uid, infos)
        
        return self.theta, self.selected, self.observation

    def load_data(self, ncatdata: Any, concept: Dict) -> Tuple[Dict, int, Dict]:
        """
        加载数据
        
        Args:
            ncatdata: NCAT数据
            concept: 概念映射
            
        Returns:
            rates, item_num, know_map
        """
        return ncatdata.data, ncatdata.num_questions, concept

    def compute_div_reward(self, concept_map: Dict, tested_questions: List[int], qid: int) -> int:
        """
        计算多样性奖励
        
        Args:
            concept_map: 概念映射
            tested_questions: 已测试的问题
            qid: 问题ID
            
        Returns:
            奖励值
        """
        concept_cnt = set()
        reward = 0
        
        # 收集已测试问题的概念
        for q in list(tested_questions):
            for c in concept_map[q]:
                concept_cnt.add(c)
        
        # 检查新问题是否包含新概念
        for c in concept_map[qid]:
            if c not in concept_cnt:
                reward = 1
                break
                
        return reward

    def check_similar_question(self, question_embedding: List[float], tested: List[Dict]) -> int:
        """
        检查问题是否与已测试问题相似
        
        Args:
            question_embedding: 问题嵌入
            tested: 已测试问题列表
            
        Returns:
            1表示相似，0表示不相似，-1表示没有类似问题
        """
        similar_list = []
        
        # 找出相似问题
        for q in tested:
            q_emb = q['question_embedding']
            
            # 计算余弦相似度
            sim = np.dot(question_embedding, q_emb) / (np.linalg.norm(question_embedding) * np.linalg.norm(q_emb))
            
            if sim > 0.85:
                similar_list.append(q)
        
        # 计算相似问题的回答正确率
        rate_sum = 0
        for i in range(len(similar_list)):
            rate = similar_list[i]['rate']
            if rate == 1:
                rate_sum += 1
                
        # 判断相似度
        if len(similar_list) != 0:
            ratio = rate_sum / len(similar_list)
        else:
            return -1
            
        return 1 if ratio > 0.5 else 0

    def load_CDM(self, name: str, data: Any, pth_path: str, config: Dict) -> Tuple[Any, Dict]:
        """
        加载认知诊断模型
        
        Args:
            name: 模型名称
            data: 数据
            pth_path: 模型路径
            config: 配置参数
            
        Returns:
            model, dataset
        """
        if name == 'IRT':
            from src.models.irt_model import IRTModel
            model = IRTModel(**config)
            model.init_model(data)
            model.adaptest_load(pth_path)
        else:
            from src.models.ncd_model import NCDModel
            model = NCDModel(**config)
            model.init_model(data)
            model.adaptest_load(pth_path)
            
        return model, data.data

    def step(self, action: int, sid: int, infos: Dict) -> Tuple:
        """
        执行步骤
        
        Args:
            action: 动作
            sid: 用户ID
            infos: 信息集
            
        Returns:
            state, rate, reward, records, done, selected, theta
        """
        # 验证动作合法性
        assert action in self.sup_rates[self.state[0][0]] and action not in self.short
        
        # 计算奖励
        reward, acc, auc, rate = self.reward(action, sid)
        
        # 判断是否结束
        if len(self.state[1]) < self.T - 1:
            done = False
        else:
            done = True
            
        # 更新状态
        self.short[action] = 1
        t = self.state[1] + [[action, reward, done]]
        
        # 记录信息
        info = {
            "ACC": acc,
            "AUC": auc,
            "rate": rate
        }
        
        # 获取问题嵌入
        question_detail = get_problem_detail([action], self.dataset_name)
        question_embedding = get_embedding(question_detail[0] if isinstance(question_detail, list) else question_detail)
        
        # 检查问题相似度
        if rate == self.check_similar_question(question_embedding, self.tested_embs):
            rate_change = 0
        else:
            rate_change = 1
            
        # 记录问题信息
        question_info = {
            "question_embedding": question_embedding,
            "rate": rate,
        }
        self.tested_embs.append(question_info)
        
        # 添加到记录
        self.state[1].append([action, reward, done, info])
        text = '正确' if rate == 1 else '错误'
        self.records.append({str(question_detail): text})
        
        # 获取能力参数
        if self.CDM == 'IRT':
            self.theta = self.model.get_theta(sid)
        else:
            self.theta = self.model.get_knowledge_status(sid)
            
        # 更新信息
        infos[len(self.state[1])].append({sid: action})
        
        # 计算奖励增量
        if self.last_ACC is not None:
            acc_add = acc - self.last_ACC
        else:
            acc_add = 0
            
        if self.last_AUC is not None:
            auc_add = auc - self.last_AUC
        else:
            auc_add = 0
            
        # 二值化奖励增量
        auc_add = 1 if auc_add > 0 else 0
        acc_add = 1 if acc_add > 0 else 0
        
        # 计算知识点覆盖奖励
        cov = self.compute_div_reward(self.know_map, self.selected, action)
        
        # 组合奖励
        reward = [auc_add, rate_change, cov]
        
        # 更新选中问题
        self.selected.append(action)
        
        # 更新候选项
        candidate_items = self.candidate_items
        for item in list(candidate_items):
            if action in self.selected and item == action:
                candidate_items.remove(item)
                
        # 更新上一次指标
        self.last_ACC = acc
        self.last_rate = rate
        self.last_AUC = auc
        
        return self.state, rate, reward, self.records, done, self.selected, self.theta

    def reward(self, action: int, sid: int) -> Tuple[float, float, float, int]:
        """
        计算奖励
        
        Args:
            action: 动作
            sid: 用户ID
            
        Returns:
            loss, acc, auc, correct
        """
        # 收集项目和正确答案
        items = [state[0] for state in self.state[1]] + [action]
        correct = [self.rates[self.state[0][0]][it] for it in items]
        
        # 应用选择
        self.tsdata.apply_selection(sid, action)
        
        # 根据模型类型更新
        if self.CDM == 'IRT':
            theta_old = self.model.get_theta(sid).copy()
            loss = self.model.adaptest_update(self.tsdata)
            theta_new = self.model.get_theta(sid)
        else:
            theta_old = self.model.get_knowledge_status(sid).copy()
            loss = self.model.adaptest_update(self.tsdata)
            theta_new = self.model.get_knowledge_status(sid)
            
        # 计算theta变化
        theta_diff = np.linalg.norm(theta_new - theta_old)
        
        # 评估模型
        result = self.model.evaluate(self.tsdata)
        acc = result['acc']
        auc = result['auc']
        
        return loss, acc, auc, correct[-1]

    def precision(self, episode: List) -> float:
        """
        计算精度
        
        Args:
            episode: 回合记录
            
        Returns:
            精度
        """
        return sum([i[1] for i in episode])