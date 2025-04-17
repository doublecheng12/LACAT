"""
自适应策略模块，实现基于LLM的自适应测试策略。
"""

import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from functools import partial

from src.agents.adaptive_agent import AdaptiveAgent
from src.dataset.cat_dataset import AdapTestDataset

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ada_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def default_converter(obj):
    """
    默认JSON转换器，处理无法序列化的类型
    
    Args:
        obj: 要转换的对象
        
    Returns:
        转换后的对象
    """
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class ADAStrategy:
    """基于LLM的自适应测试策略"""
    
    def __init__(
        self,
        llm_name: str = "gpt-3.5-turbo",
        max_steps: int = 5,
        threshold: float = 0.0,
        result_path: str = "results/ada_results.json",
        log_path: str = "logs/ada_strategy.log"
    ):
        """
        初始化自适应策略
        
        Args:
            llm_name: 语言模型名称
            max_steps: 最大步数
            threshold: 奖励阈值
            result_path: 结果保存路径
            log_path: 日志路径
        """
        self.llm_name = llm_name
        self.max_steps = max_steps
        self.threshold = threshold
        self.result_path = result_path
        self.log_path = log_path
        
        logger.info(f"初始化自适应策略，使用模型: {llm_name}")

    @property
    def name(self) -> str:
        """
        获取策略名称
        
        Returns:
            策略名称
        """
        return 'ADA'

    def adaptest_select(
        self, 
        adaptest_data: AdapTestDataset,
        concept_map: Dict,
        config: Dict,
        test_length: int,
        seed: int = 42,
        verbose: bool = True
    ) -> List[int]:
        """
        执行自适应测试选择
        
        Args:
            adaptest_data: 测试数据集
            concept_map: 概念映射
            config: 配置参数
            test_length: 测试长度
            seed: 随机种子
            verbose: 是否显示进度
            
        Returns:
            已使用的问题ID列表
        """
        # 设置随机种子
        np.random.seed(seed)
        
        used_actions = []
        infos = {item: [] for item in range(1, test_length + 1)}
        
        # 遍历所有学生
        iterator = tqdm(range(adaptest_data.num_students)) if verbose else range(adaptest_data.num_students)
        
        for sid in iterator:
            # 创建自适应代理
            agent = AdaptiveAgent(
                name=f'ada_{sid}',
                data=adaptest_data,
                concept_map=concept_map,
                config=config,
                test_length=test_length,
                llm_name=self.llm_name,
                max_steps=self.max_steps,
                threshold=self.threshold,
                log_path=self.log_path
            )
            
            # 运行代理
            agent.run(sid, used_actions, infos)
        
        # 计算和记录结果
        self._log_results(infos, test_length)
        
        return used_actions

    def _log_results(self, infos: Dict[int, List], test_length: int) -> None:
        """
        记录结果
        
        Args:
            infos: 信息收集字典
            test_length: 测试长度
        """
        # 计算ACC和AUC
        results = {}
        for item in range(3, test_length + 1, 2):
            if item in infos and infos[item]:
                acc = np.mean([i.get("ACC", 0) for i in infos[item]])
                auc = np.mean([i.get("AUC", 0) for i in infos[item]])
                results[f"{item}_ACC"] = round(float(acc), 4)
                results[f"{item}_AUC"] = round(float(auc), 4)
                logger.info(f"测试长度 {item} - ACC: {acc:.4f}, AUC: {auc:.4f}")
        
        # 保存结果
        try:
            with open(self.result_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {"infos": infos, "results": results},
                    f, 
                    default=default_converter,
                    indent=2,
                    ensure_ascii=False
                )
            logger.info(f"结果已保存到: {self.result_path}")
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}", exc_info=True)