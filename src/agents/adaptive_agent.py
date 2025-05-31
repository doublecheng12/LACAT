"""
自适应代理模块，实现基于LLM的计算机自适应测试代理。
"""

import random
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch

from src.models.gpt import LLM
from src.environment.env import Environment
from src.llms.generator import Generator
from src.llms.first_try import FirstTry
from src.llms.longterm_memory import LongTermMemory
from src.llms.short_memory import ShortTermMemory
from src.llms.critic import Critic
from src.utils.data_utils import get_problem_detail, get_correct

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/adaptive_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdaptiveAgent:
    """自适应学习代理，基于LLM实现计算机自适应测试"""
    
    def __init__(
        self,
        name: str,
        data: Any,
        concept_map: Dict,
        config: Dict,
        test_length: int,
        llm_name: str,
        max_steps: int,
        threshold: float = 0.0,
        log_path: Optional[str] = None,
    ) -> None:
        """
        初始化自适应代理
        
        Args:
            name: 代理名称
            data: 测试数据
            concept_map: 概念映射
            config: 配置参数
            test_length: 测试长度
            llm_name: 语言模型名称
            max_steps: 最大步数
            threshold: 奖励阈值
            log_path: 日志路径
        """
        self.name = name
        self.max_steps = max_steps
        self.threshold = threshold
        self.theta_avg = 0
        self.env = Environment(data=data, concept_map=concept_map, config=config, T=test_length)
        self.llm_name = llm_name
        self.records = []
        self.feedback = ""
        self.log_path = log_path or "results/agent_logs.jsonl"
        
        # 初始化LLM组件
        logger.info(f"初始化自适应代理 {name}，使用模型 {llm_name}")
        self.llm = LLM(self.llm_name)
        self.generator = Generator(llm=self.llm)
        self.longterm_memory = LongTermMemory(llm=self.llm)
        self.shortterm_memory = ShortTermMemory(llm=self.llm)
        self.first_try = FirstTry(llm=self.llm)
        self.profile = ""
        self.critic = Critic(llm=self.llm)

        
        # 运行计数器和状态
        self.run_count = 0
        self.selected = []
        self.last_question = ""
        self.strength = ""
        self.weakness = ""
        self.answer = ""
        
        logger.info(f"自适应代理 {name} 初始化完成")

    def reset(self) -> Tuple[Any, List]:
        """
        重置环境和代理状态
        
        Returns:
            环境状态和记录
        """
        self.env.reset()
        self.records = []
        self.feedback = ""
        self.profile = ""
        self.selected = []
        
        return self.env.state, self.env.records

    def step(self, sid: int, action: int, infos: Dict[int, List]) -> None:
        """
        执行一步操作
        
        Args:
            sid: 学生ID
            action: 选择的问题ID
            infos: 信息收集字典
        """
        try:
            _, rate, observation, self.records, done, self.selected, self.theta = self.env.step(action, sid, infos)
            
            # 计算加权奖励
            reward = 0
            weights = [0.5, 0.3, 0.2]
            for i, num in enumerate(observation):
                reward += num * weights[i]
                
            # 记录步骤信息
            logger.info(f"步骤完成: 学生 {sid}, 问题 {action}, 奖励 {reward:.4f}")
            
            # 如果奖励低于阈值，生成反馈
            if reward < self.threshold:
                profile = self.profile
                question_detail = get_problem_detail([action], self.env.dataset_name)
                self.feedback = self.critic.answer(profile, question_detail)
                logger.info(f"生成反馈: {self.feedback}")
        
        except Exception as e:
            logger.error(f"步骤执行错误: {str(e)}", exc_info=True)

    def run(self, sid: int, used_actions: List[int], infos: Dict[int, List], reset: bool = True) -> None:
        """
        运行代理完成一个完整的评估
        
        Args:
            sid: 学生ID
            used_actions: 已使用的问题ID列表
            infos: 信息收集字典
            reset: 是否重置环境
        """
        self.run_count += 1
        start_time = datetime.now()
        logger.info(f"开始为学生 {sid} 运行评估 #{self.run_count}, 时间: {start_time}")
        
        try:
            # 初始化第一个问题
            candidate = list(self.env.sup_rates[sid].keys())
            first_question_details = get_problem_detail(candidate, self.env.dataset_name)
            action = self.first_try.answer(first_question_details)
            
            try:
                action = candidate[int(action)]
            except (ValueError, IndexError):
                logger.warning(f"无法解析第一个问题选择: {action}，随机选择")
                action = random.choice(candidate)
            
            # 重置环境
            self.theta, self.selected, self.theta_avg = self.env.reset_with_users(sid, action, infos)
            
            # 主循环
            done = False
            iteration = 0
            
            while not done:
                iteration += 1
                logger.info(f"开始迭代 #{iteration}, 学生 {sid}")
                
                # 收集已回答的问题
                right_questions = []
                wrong_questions = []
                knowledge_hint = ""
                
                # 分类问题
                for qid in self.selected:
                    if get_correct(sid, qid, self.env.dataset_name) == 1:
                        right_questions.append(qid)
                    else:
                        wrong_questions.append(qid)
                
                # 获取最近一次问题和回答
                self.last_question = get_problem_detail([self.selected[-1]], self.env.dataset_name)
                if get_correct(sid, self.selected[-1], self.env.dataset_name) == 1:
                    self.last_question = str(self.last_question) + ":正确"
                else:
                    self.last_question = str(self.last_question) + ":错误"
                
                # 获取长期记忆（学生优势和劣势）
                right_details = get_problem_detail(right_questions, self.env.dataset_name)
                wrong_details = get_problem_detail(wrong_questions, self.env.dataset_name)
                self.strength, self.weakness = self.longterm_memory.answer(right_details, wrong_details)
                
                # 构建学生画像
                self.profile = str(self.strength) + str(self.weakness)
                
                # 获取短期记忆（对最近问题的推理）
                self.answer = self.shortterm_memory.answer(self.profile, self.last_question)
                
                # 获取候选问题
                candidate_items = []
                all_items = self.env.candidate_items
                for item in all_items:
                    if item not in self.selected:
                        candidate_items.append(item)
                
                # 设置最多推荐15个候选问题
                self.candidate = candidate_items[:15] if len(candidate_items) > 15 else candidate_items
                
                # 使用反馈作为知识提示
                knowledge_hint = self.feedback
                self.feedback = ''
                
                # 使用CAT模型选择下一个问题
                cat_agent = self._create_cat_agent(knowledge_hint)
                output = cat_agent.run()
                
                action = output.get('question_index')
                reason = output.get('reason', "")
                
                try:
                    action = self.candidate[int(action)]
                except (ValueError, IndexError, TypeError):
                    logger.warning(f"无法解析问题选择: {action}，随机选择")
                    action = random.choice(self.candidate)
                
                # 记录解释到日志
                self._log_explanation(sid, action, reason)
                
                # 执行步骤
                _, rate, self.observation, self.records, done, self.selected, self.theta = self.env.step(action, sid, infos)
                
                # 计算奖励
                reward = 0
                weights = [0.5, 0.3, 0.2]
                for i, num in enumerate(self.observation):
                    reward += num * weights[i]
                
                # 如果奖励低于阈值，生成反馈
                try:
                    if reward < self.threshold:
                        self.feedback = self.critic.answer(self.profile, get_problem_detail([action], self.env.dataset_name))
                except Exception as e:
                    logger.error(f"生成反馈出错: {str(e)}")
            
            # 记录已使用的问题
            used_actions.extend(self.selected)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"完成学生 {sid} 的评估，用时: {duration:.2f}秒，选择了 {len(self.selected)} 个问题")
        
        except Exception as e:
            logger.error(f"运行过程中出错: {str(e)}", exc_info=True)
    
    def _create_cat_agent(self, knowledge_hint: str) -> Any:
        """
        创建CAT代理进行问题选择
        
        Args:
            knowledge_hint: 知识提示
            
        Returns:
            CAT代理
        """
        from src.cat.cat_agent import CATAgent
        
        return CATAgent(
            llm_name=self.llm_name, 
            strength=self.strength, 
            weakness=self.weakness,
            last_question=self.last_question, 
            answer=self.answer,
            dataset_name=self.env.dataset_name, 
            candidate=self.candidate,
            knowledge_hint=knowledge_hint,
            theta=self.theta
        )
    
    def _log_explanation(self, sid: int, action: int, reason: str) -> None:
        """
        记录解释到日志文件
        
        Args:
            sid: 学生ID
            action: 选择的问题ID
            reason: 选择原因
        """
        if not reason:
            return
            
        try:
            explanation_dict = {
                'timestamp': datetime.now().isoformat(),
                'student_id': sid,
                'records': str(self.records),
                'profile': str(self.profile),
                'selection': str(get_problem_detail([action], self.env.dataset_name)),
                'reason': reason
            }
            
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(explanation_dict, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"记录解释出错: {str(e)}")
