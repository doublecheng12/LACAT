"""
生成器模块，用于选择最适合学生当前水平的问题。
"""

import re
import logging
from typing import Dict, Tuple, List, Any, Union, Optional

from src.models.gpt import LLM
from src.prompts.prompt_templates import PromptTemplates
from src.utils.data_utils import vector2description

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Generator:
    """生成器模块，用于选择最适合学生当前水平的问题"""
    
    def __init__(self, llm: LLM):
        """
        初始化生成器
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        logger.info("初始化生成器模块")

    def answer(self, records: List[Dict], candidate_questions: List[str]) -> Tuple[str, str]:
        """
        基于学生记录选择问题
        
        Args:
            records: 学生答题记录
            candidate_questions: 候选问题列表
            
        Returns:
            选择原因和选择的问题索引
        """
        logger.info("基于学生记录选择问题")
        
        try:
            # 生成提示
            ge_prompt = PromptTemplates.GE_PROMPT.format(
                records=str(records),
                candidate_questions=str(candidate_questions)
            )
            
            # 获取LLM输出
            output = self.llm.get_output(
                prompt=ge_prompt, 
                system_prompt="现在你是教育推荐专家，根据学生的做题序列和当前水平，从候选题库中选择最适合的习题"
            )
            
            # 解析输出
            try:
                # 提取JSON部分
                json_pattern = r'\{.*?\}'
                json_matches = re.findall(json_pattern, output, re.DOTALL)
                
                if not json_matches:
                    logger.warning(f"未找到JSON格式输出: {output}")
                    return str(output), "0"
                
                # 解析第一个匹配的JSON
                result = eval(json_matches[0])
                logger.info(f"选择结果: {result}")
                
                return str(result['reason']), str(result['question_index'])
                
            except Exception as e:
                logger.error(f"解析错误: {str(e)}\n原始输出: {output}")
                return str(output), str(output)
                
        except Exception as e:
            logger.error(f"选择问题出错: {str(e)}", exc_info=True)
            return str(e), "0"

    def cog_answer(
        self, 
        strength: str,
        weakness: str,
        last_question: str,
        answer: str,
        candidate: List[str],
        knowledge_hint: str = ""
    ) -> Dict[str, Any]:
        """
        基于认知诊断选择问题
        
        Args:
            strength: 学生优势
            weakness: 学生劣势
            last_question: 上一个问题
            answer: 上一个问题的回答
            candidate: 候选问题列表
            knowledge_hint: 知识提示
            
        Returns:
            包含选择原因和问题索引的字典
        """
        logger.info("基于认知诊断选择问题")
        
        try:
            # 生成提示
            rge_prompt = PromptTemplates.RGE_PROMPT.format(
                strength=strength,
                weakness=weakness, 
                last_question=last_question,
                answer=answer,
                candidate_questions=candidate
            )
            
            # 添加知识提示
            if knowledge_hint:
                rge_prompt += "\nknowledge_hint: " + knowledge_hint
            
            # 获取LLM输出
            output = self.llm.get_output(
                prompt=rge_prompt, 
                system_prompt="现在你是教育推荐专家，根据学生当前水平，从候选题库中选择最适合的习题"
            )
            
            # 解析输出
            try:
                # 提取JSON部分
                json_pattern = r'\{.*?\}'
                json_matches = re.findall(json_pattern, output, re.DOTALL)
                
                if not json_matches:
                    logger.warning(f"未找到JSON格式输出: {output}")
                    return {"question_index": 0, "reason": "解析错误，请查看日志"}
                
                # 解析第一个匹配的JSON
                result = eval(json_matches[0])
                logger.info(f"选择结果: {result}")
                
                return result
                
            except Exception as e:
                logger.error(f"解析错误: {str(e)}\n原始输出: {output}")
                return {"question_index": 0, "reason": f"解析错误: {str(e)}"}
                
        except Exception as e:
            logger.error(f"认知诊断选择出错: {str(e)}", exc_info=True)
            return {"question_index": 0, "reason": f"选择出错: {str(e)}"}

    def theta_answer(
        self, 
        theta: Union[float, List[float]],
        candidate: List[str],
        last_question: str,
        answer: str,
        records: List[Dict],
        knowledge_hint: str = ""
    ) -> Union[Dict[str, Any], Tuple[str, str]]:
        """
        基于能力参数选择问题
        
        Args:
            theta: 能力参数
            candidate: 候选问题列表
            last_question: 上一个问题
            answer: 上一个问题的回答
            records: 学生记录
            knowledge_hint: 知识提示
            
        Returns:
            包含选择原因和问题索引的字典或元组
        """
        logger.info("基于能力参数选择问题")
        
        try:
            # 根据能力参数类型选择不同的提示模板
            if isinstance(theta, (int, float)) or (isinstance(theta, list) and len(theta) == 1):
                # 使用IRT模型的theta值
                theta_prompt = PromptTemplates.THETA_PROMPT.format(
                    theta=theta,
                    candidate_questions=candidate,
                    last_question=last_question,
                    answer=answer
                )
                
                # 添加知识提示
                if knowledge_hint:
                    theta_prompt += "\nknowledge_hint: " + knowledge_hint
                
                output = self.llm.get_output(
                    prompt=theta_prompt,
                    system_prompt="现在你是教育推荐专家，根据学生当前水平，从候选题库中选择最适合的习题"
                )
                
                try:
                    # 提取JSON部分
                    json_pattern = r'\{.*?\}'
                    json_matches = re.findall(json_pattern, output, re.DOTALL)
                    
                    if not json_matches:
                        logger.warning(f"未找到JSON格式输出: {output}")
                        return {"question_index": 0, "reason": "解析错误，请查看日志"}
                    
                    # 解析第一个匹配的JSON
                    result = eval(json_matches[0])
                    logger.info(f"选择结果: {result}")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"解析错误: {str(e)}\n原始输出: {output}")
                    return {"question_index": 0, "reason": f"解析错误: {str(e)}"}
                    
            else:
                # 使用向量形式的能力参数（NCD模型）
                from src.utils.data_utils import vector2description
                from src.llms.monitor import Monitor
                
                # 将向量转换为描述
                theta_desc = vector2description(theta)
                
                # 使用监视器分析学生记录
                monitor = Monitor(self.llm)
                theta_desc = monitor.answer(theta_desc, records)
                
                # 生成提示
                theta_prompt = PromptTemplates.NCD_THETA_PROMPT.format(
                    theta=theta_desc,
                    candidate_questions=candidate,
                    last_question=last_question,
                    answer=answer
                )
                
                # 添加知识提示
                if knowledge_hint:
                    theta_prompt += "\nknowledge_hint: " + knowledge_hint
                
                output = self.llm.get_output(
                    prompt=theta_prompt,
                    system_prompt="现在你是教育推荐专家，根据学生当前水平，从候选题库中选择最适合的习题"
                )
                
                try:
                    # 提取JSON部分
                    json_pattern = r'\{.*?\}'
                    json_matches = re.findall(json_pattern, output, re.DOTALL)
                    
                    if not json_matches:
                        logger.warning(f"未找到JSON格式输出: {output}")
                        return str(output), "0"
                    
                    # 解析第一个匹配的JSON
                    result = eval(json_matches[0])
                    logger.info(f"选择结果: {result}")
                    
                    return str(result['reason']), str(result['question_index'])
                    
                except Exception as e:
                    logger.error(f"解析错误: {str(e)}\n原始输出: {output}")
                    return str(output), "0"
                    
        except Exception as e:
            logger.error(f"基于能力参数选择出错: {str(e)}", exc_info=True)
            return {"question_index": 0, "reason": f"选择出错: {str(e)}"}