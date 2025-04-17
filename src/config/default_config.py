"""
默认配置文件，提供系统的默认参数配置。
"""

# 通用配置
DEFAULT_CONFIG = {
    # 系统配置
    "system": {
        "seed": 42,
        "device": "cuda:0",
        "log_level": "INFO",
        "result_dir": "results",
        "checkpoint_dir": "ckpt",
        "data_dir": "data"
    },
    
    # 模型配置
    "model": {
        "irt": {
            "lr": 0.003,
            "batch_size": 32,
            "epochs": 30,
            "device": "cuda:0"
        },
        "ncd": {
            "lr": 0.002,
            "batch_size": 32,
            "epochs": 30,
            "knowledge_dim": 16,
            "device": "cuda:0"
        }
    },
    
    # 自适应测试配置
    "adaptest": {
        "test_length": 10,
        "threshold": 0.0,
        "max_steps": 5
    },
    
    # 语言模型配置
    "llm": {
        "name": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 1000,
        "api_key_env": "OPENAI_API_KEY",
        "api_base_env": "OPENAI_API_BASE"
    },
    
    # 数据集配置
    "dataset": {
        "name": "moocradar",
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "batch_size": 32,
        "shuffle": True
    }
}