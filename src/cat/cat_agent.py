"""
CAT代理模块，实现认知自适应测试代理。
"""

import logging
from typing import Dict, List, Any, Optional
from src.models.gpt import LLM
from src.llms.generator import Generator
from src.utils.data_utils import get_problem_detail

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/cat_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CATAgent:
    """计算机自适应测试代理，负责选择最适合学生的下一个问题"""
    
    def __init__(
        self,
        llm_name: str,
        strength: str = "",
        weakness: str = "",
        last_question: str = "",
        answer: str = "",
        candidate: List[int] = None,
        records: str = "",
        knowledge_hint: str = "",
        theta: Any = None,
        dataset_name: str = ""
    ):
        """
        初始化CAT代理
        
        Args:
            llm_name: 语言模型名称
            strength: 学生优势
            weakness: 学生劣势
            last_question: 上一个问题
            answer: 上一个问题的回答
            candidate: 候选问题ID列表
            records: 学生记录
            knowledge_hint: 知识提示
            theta: 能力参数
            dataset_name: 数据集名称
        """
        self.llm_name = llm_name
        self.strength = strength
        self.weakness = weakness
        self.last_question = last_question
        self.answer = answer
        self.candidate = candidate or []
        self.theta = theta
        self.dataset_name = dataset_name
        self.records = records
        self.knowledge_hint = knowledge_hint
        
        # 初始化LLM
        logger.info(f"初始化CAT代理，使用模型: {llm_name}")
        self.llm = LLM(llm_name)
        self.generator = Generator(llm=self.llm)

    def run(self) -> Dict[str, Any]:
        """
        运行CAT代理选择下一个问题
        
        Returns:
            包含选择的问题索引和原因的字典
        """
        logger.info("运行CAT代理选择问题")
        
        try:
            # 获取候选问题的详情
            candidate_details = get_problem_detail(self.candidate, self.dataset_name)
            
            # 使用生成器选择问题
            output = self.generator.cog_answer(
                strength=self.strength, 
                weakness=self.weakness,
                last_question=self.last_question, 
                answer=self.answer,
                candidate=candidate_details,
                knowledge_hint=self.knowledge_hint
            )
            
            logger.info(f"CAT代理选择完成: {output}")
            return output
            
        except Exception as e:
            logger.error(f"CAT代理运行出错: {str(e)}", exc_info=True)
            # 返回一个默认值表示错误
            return {
                "error": str(e),
                "question_index": 0,
                "reason": "运行出错，默认选择第一个问题"
            }