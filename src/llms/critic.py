"""
评论家模块，用于分析选择的问题是否适合学生。
"""

import re
import logging
from typing import Optional, Dict, Any

from src.models.gpt import LLM
from src.prompts.prompt_templates import PromptTemplates

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/critic.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Critic:
    """评论家模块，用于分析选择的问题是否适合学生"""
    
    def __init__(self, llm: LLM):
        """
        初始化评论家模块
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        logger.info("初始化评论家模块")

    def answer(self, profile: str, last_question: str) -> str:
        """
        分析选择的问题是否适合学生
        
        Args:
            profile: 学生画像
            last_question: 最近选择的问题
            
        Returns:
            反馈意见
        """
        logger.info("开始分析问题")
        
        try:
            # 生成提示
            critic_prompt = PromptTemplates.CRITIC_PROMPT.format(
                profile=str(profile),
                exercise=str(last_question)
            )
            
            # 获取LLM输出
            output = self.llm.get_output(
                prompt=critic_prompt, 
                system_prompt="now you are an expert in education analysis"
            )
            
            # 解析输出
            try:
                # 移除可能的Markdown格式
                cleaned_output = re.sub(r'^\s*```json\s*\n|\n\s*```\s*$', '', output, flags=re.MULTILINE)
                
                # 解析JSON
                result = eval(cleaned_output)
                logger.info(f"分析结果: {result}")
                
                return str(result['feedback'])
                
            except Exception as e:
                logger.error(f"解析错误: {str(e)}\n原始输出: {output}")
                return str(output)
                
        except Exception as e:
            logger.error(f"分析问题出错: {str(e)}", exc_info=True)
            return f"分析出错: {str(e)}"