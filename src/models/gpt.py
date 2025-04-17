import os
import logging
from typing import Optional, Dict, Any, Union
import time
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api_calls.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 默认API参数
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
MAX_RETRY_ATTEMPTS = 5
RETRY_DELAY = 2  # 秒


class LLM:
    """语言模型接口类，用于与不同的LLM服务提供商交互"""
    
    def __init__(
        self, 
        model_name: str = "gpt-3.5-turbo", 
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_retries: int = MAX_RETRY_ATTEMPTS
    ):
        """
        初始化LLM接口
        
        Args:
            model_name: 模型名称
            api_key: API密钥，如果为None则使用环境变量
            api_base: API基础URL，如果为None则使用环境变量
            max_retries: 最大重试次数
        """
        self.model_name = model_name
        self.api_key = api_key or DEFAULT_API_KEY
        self.api_base = api_base or DEFAULT_API_BASE
        self.max_retries = max_retries
        
        if not self.api_key:
            raise ValueError("API密钥未设置。请设置OPENAI_API_KEY环境变量或直接提供api_key参数")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        logger.info(f"初始化LLM接口，模型: {model_name}")
    
    def get_output(self, prompt: str, system_prompt: str = "你是一名教育专家") -> str:
        """
        从语言模型获取输出
        
        Args:
            prompt: 提示文本
            system_prompt: 系统提示
            
        Returns:
            模型生成的回复
        """
        if "gpt" in self.model_name.lower():
            return self._get_openai_output(prompt, system_prompt)
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")
    
    def _get_openai_output(self, prompt: str, system_prompt: str) -> str:
        """
        调用OpenAI API获取输出，包含重试逻辑
        
        Args:
            prompt: 提示文本
            system_prompt: 系统提示
            
        Returns:
            OpenAI API的响应文本
        """
        attempt = 0
        last_exception = None
        
        while attempt < self.max_retries:
            try:
                logger.debug(f"调用OpenAI API，尝试 #{attempt+1}")
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                
                duration = time.time() - start_time
                logger.info(f"API调用成功，耗时: {duration:.2f}秒")
                
                output = response.choices[0].message.content
                return output
                
            except Exception as e:
                attempt += 1
                last_exception = e
                logger.warning(f"API调用失败 (尝试 {attempt}/{self.max_retries}): {str(e)}")
                
                if attempt < self.max_retries:
                    # 使用指数退避策略
                    sleep_time = RETRY_DELAY * (2 ** (attempt - 1))
                    logger.info(f"等待 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
        
        # 所有重试都失败了
        logger.error(f"所有API调用尝试均失败: {str(last_exception)}")
        return '{"error": "API调用失败", "details": "' + str(last_exception).replace('"', '\\"') + '"}'