"""
短期记忆模块，用于分析学生最近的答题情况。
"""

import re
import logging
from typing import Optional

from src.models.gpt import LLM
from src.prompts.prompt_templates import PromptTemplates

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/short_memory.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ShortTermMemory:
    """短期记忆模块，用于分析学生最近的答题情况"""
    
    def __init__(self, llm: LLM):
        """
        初始化短期记忆模块
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        logger.info("初始化短期记忆模块")

    def answer(self, profile: str, last_question: str) -> str:
        """
        分析学生最近的答题情况
        
        Args:
            profile: 学生画像
            last_question: 最近的问题
            
        Returns:
            分析结果
        """
        logger.info("开始分析最近答题情况")
        
        try:
            # 生成提示
            short_term_memory = PromptTemplates.SHORT_MEMORY_PROMPT.format(
                profile=str(profile),
                last_question=str(last_question)
            )
            
            # 获取LLM输出
            output = self.llm.get_output(
                prompt=short_term_memory, 
                system_prompt="now you are an expert in education analysis"
            )
            
            # 解析输出
            try:
                # 移除可能的Markdown格式
                cleaned_output = re.sub(r'^\s*```json\s*\n|\n\s*```\s*$', '', output, flags=re.MULTILINE)
                
                # 解析JSON
                result = eval(cleaned_output)
                logger.info(f"分析结果: {result}")
                
                return str(result['thought'])
                
            except Exception as e:
                logger.error(f"解析错误: {str(e)}\n原始输出: {output}")
                return str(output)
                
        except Exception as e:
            logger.error(f"分析最近答题情况出错: {str(e)}", exc_info=True)
            return "分析出错"