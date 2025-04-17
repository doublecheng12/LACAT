"""
长期记忆模块，用于分析学生的优势和劣势。
"""

import re
import logging
from typing import Tuple, List, Dict, Any, Optional

from src.models.gpt import LLM
from src.prompts.prompt_templates import PromptTemplates

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/longterm_memory.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LongTermMemory:
    """长期记忆模块，用于分析学生的优势和劣势"""
    
    def __init__(self, llm: LLM):
        """
        初始化长期记忆模块
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        logger.info("初始化长期记忆模块")

    def answer(self, right_ids: List[str], wrong_ids: List[str]) -> Tuple[str, str]:
        """
        分析学生的优势和劣势
        
        Args:
            right_ids: 正确回答的问题列表
            wrong_ids: 错误回答的问题列表
            
        Returns:
            学生的优势和劣势
        """
        logger.info("开始分析学生优势和劣势")
        
        try:
            # 生成提示
            long_term_memory = PromptTemplates.LONGTERM_MEMORY_PROMPT.format(
                right_answer_records=str(right_ids),
                wrong_answer_records=str(wrong_ids)
            )
            
            # 获取LLM输出
            output = self.llm.get_output(
                prompt=long_term_memory, 
                system_prompt="现在你是教育分析专家，请从学生所掌握或欠缺的知识点角度，分析学生的优势和不足"
            )
            
            # 解析输出
            try:
                # 移除可能的Markdown格式
                cleaned_output = re.sub(r'^\s*```json\s*\n|\n\s*```\s*$', '', output, flags=re.MULTILINE)
                
                # 解析JSON
                result = eval(cleaned_output)
                logger.info(f"分析结果: {result}")
                
                return str(result['strength']), str(result['weakness'])
                
            except Exception as e:
                logger.error(f"解析错误: {str(e)}\n原始输出: {output}")
                return str(output), ""
                
        except Exception as e:
            logger.error(f"分析优势劣势出错: {str(e)}", exc_info=True)
            return "无法分析优势", "无法分析劣势"