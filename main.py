"""
主模块，包含程序入口点和命令行参数解析。
"""

import os
import argparse
import logging
import yaml
import torch
import numpy as np
import random
from datetime import datetime
from typing import Dict, Any, Optional

from src.dataset.cat_dataset import AdapTestDataset
from src.strategies.ada_strategy import ADAStrategy
from src.utils.data_utils import load_json_file

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置加载成功: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置失败: {str(e)}", exc_info=True)
        from config.default_config import DEFAULT_CONFIG
        logger.info("使用默认配置")
        return DEFAULT_CONFIG


def set_seed(seed: int) -> None:
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"随机种子设置为: {seed}")


def prepare_dirs(config: Dict[str, Any]) -> None:
    """
    准备目录
    
    Args:
        config: 配置字典
    """
    # 创建结果和检查点目录
    os.makedirs(config["system"]["result_dir"], exist_ok=True)
    os.makedirs(config["system"]["checkpoint_dir"], exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 创建数据集特定目录
    dataset_dir = os.path.join(config["system"]["result_dir"], config["dataset"]["name"])
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(dataset_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    logger.info(f"创建运行目录: {run_dir}")
    return run_dir


def load_dataset(config: Dict[str, Any]) -> Tuple[AdapTestDataset, Dict]:
    """
    加载数据集
    
    Args:
        config: 配置字典
        
    Returns:
        数据集和概念映射
    """
    dataset_name = config["dataset"]["name"]
    data_dir = os.path.join(config["system"]["data_dir"], dataset_name)
    
    try:
        # 加载概念映射
        concept_map_path = os.path.join(data_dir, "concept_map.json")
        concept_map = load_json_file(concept_map_path)
        
        # 创建数据集
        dataset = AdapTestDataset(
            data_dir=data_dir,
            batch_size=config["dataset"]["batch_size"],
            train_ratio=config["dataset"]["train_ratio"],
            val_ratio=config["dataset"]["val_ratio"],
            test_ratio=config["dataset"]["test_ratio"],
            shuffle=config["dataset"]["shuffle"]
        )
        
        logger.info(f"数据集 {dataset_name} 加载成功")
        return dataset, concept_map
        
    except Exception as e:
        logger.error(f"加载数据集失败: {str(e)}", exc_info=True)
        raise


def run_adaptest(
    dataset: AdapTestDataset,
    concept_map: Dict,
    config: Dict[str, Any],
    run_dir: str
) -> None:
    """
    运行自适应测试
    
    Args:
        dataset: 数据集
        concept_map: 概念映射
        config: 配置字典
        run_dir: 运行目录
    """
    # 创建策略
    strategy = ADAStrategy(
        llm_name=config["llm"]["name"],
        max_steps=config["adaptest"]["max_steps"],
        threshold=config["adaptest"]["threshold"],
        result_path=os.path.join(run_dir, "results.json"),
        log_path=os.path.join(run_dir, "strategy.log")
    )
    
    # 模型配置
    model_config = config["model"]["irt"]
    
    # 运行策略
    logger.info(f"开始运行自适应测试策略: {strategy.name}")
    used_actions = strategy.adaptest_select(
        adaptest_data=dataset,
        concept_map=concept_map,
        config=model_config,
        test_length=config["adaptest"]["test_length"],
        seed=config["system"]["seed"],
        verbose=True
    )
    
    # 保存结果
    result_file = os.path.join(run_dir, "used_actions.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({"used_actions": used_actions}, f, indent=2)
    
    logger.info(f"自适应测试完成，结果保存在: {run_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于LLM的自适应测试系统")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--dataset", type=str, help="数据集名称，覆盖配置文件")
    parser.add_argument("--model", type=str, help="模型名称，覆盖配置文件")
    parser.add_argument("--test_length", type=int, help="测试长度，覆盖配置文件")
    parser.add_argument("--seed", type=int, help="随机种子，覆盖配置文件")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 更新配置
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.model:
        config["llm"]["name"] = args.model
    if args.test_length:
        config["adaptest"]["test_length"] = args.test_length
    if args.seed:
        config["system"]["seed"] = args.seed
    
    # 设置随机种子
    set_seed(config["system"]["seed"])
    
    # 准备目录
    run_dir = prepare_dirs(config)
    
    # 保存配置
    config_file = os.path.join(run_dir, "config.yaml")
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 加载数据集
    dataset, concept_map = load_dataset(config)
    
    # 运行自适应测试
    run_adaptest(dataset, concept_map, config, run_dir)


if __name__ == "__main__":
    main()