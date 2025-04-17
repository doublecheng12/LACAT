"""
首次尝试模块，用于选择第一个问题。
"""

import re
import logging
import random
from typing import Any, Union, List, Optional

from src.models.gpt import LLM
from src.prompts.prompt_templates import PromptTemplates

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/first_try.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FirstTry:
    """首次尝试模块，用于选择第一个问题"""
    
    def __init__(self, llm: LLM):
        """
        初始化首次尝试模块
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        logger.info("初始化首次尝试模块")

    def answer(self, candidate_questions: List[str]) -> str:
        """
        从候选问题中选择第一个问题
        
        Args:
            candidate_questions: 候选问题列表
            
        Returns:
            选择的问题索引
        """
        logger.info("开始选择首个问题")
        
        try:
            # 生成提示
            ge_prompt = PromptTemplates.FIRST_PROMPT.format(
                candidate_questions=str(candidate_questions)
            )
            
            # 获取LLM输出
            output = self.llm.get_output(
                prompt=ge_prompt, 
                system_prompt="现在你是教育推荐专家，从候选题库中选择一道最具有代表性的习题"
            )
            
            # 解析输出
            try:
                # 提取JSON部分
                json_pattern = r'\{.*?\}'
                json_matches = re.findall(json_pattern, output, re.DOTALL)
                
                if not json_matches:
                    logger.warning(f"未找到JSON格式输出: {output}")
                    return "0"  # 默认选择第一个问题
                
                # 解析第一个匹配的JSON
                result = eval(json_matches[0])
                logger.info(f"选择结果: {result}")
                
                return str(result['question_index'])
                
            except Exception as e:
                logger.error(f"解析错误: {str(e)}\n原始输出: {output}")
                # 默认返回第一个问题的索引
                return "0"
                
        except Exception as e:
            logger.error(f"选择首个问题出错: {str(e)}", exc_info=True)
            return "0"  # 出错时默认选择第一个问题