"""
数据处理工具模块，处理数据加载、转换和预处理等功能。
"""

import json
import os
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import torch
from openai import OpenAI
from functools import lru_cache
import faiss

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_utils.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_json_file(file_path: str) -> Dict:
    """
    加载JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        加载的JSON对象
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载JSON文件失败: {file_path}, 错误: {str(e)}")
        raise


@lru_cache(maxsize=128)
def get_problem_detail(sids: Union[List[int], Tuple[int]], dataset_name: str = 'ifly') -> List[str]:
    """
    获取问题详情
    
    Args:
        sids: 问题ID列表
        dataset_name: 数据集名称
        
    Returns:
        问题详情列表
    """
    sids = list(sids) if isinstance(sids, tuple) else sids
    
    data_path = os.path.join('data', dataset_name)
    
    try:
        questions = []
        
        if 'ifly' in dataset_name:
            # 加载相关文件
            question_map = load_json_file(os.path.join(data_path, 'select_problem_detail.json'))
            id2name = load_json_file(os.path.join(data_path, 'id2name.json'))
            question_map_reverse = load_json_file(os.path.join(data_path, 'reverse_map/question_map_reverse.json'))
            concept_map = load_json_file(os.path.join(data_path, 'concept_map.json'))
            knowledge_map_reverse = load_json_file(os.path.join(data_path, 'reverse_map/knowledge_map_reverse.json'))
            
            # 构建问题详情
            formatted_questions = ""
            for i, sid in enumerate(sids):
                sid_str = str(sid)
                # 获取题目详情
                detail = (f"{i}. {question_map[question_map_reverse[sid_str]]['detail']}\n")
                formatted_questions += detail
                
            return formatted_questions
            
        elif 'moocradar' in dataset_name:
            # 加载相关文件
            question_map = load_json_file(os.path.join(data_path, 'problem_detail.json'))
            question_map_reverse = load_json_file(os.path.join(data_path, 'reverse_map/question_map_reverse.json'))
            
            # 构建问题详情
            for sid in sids:
                sid_str = str(sid)
                questions.append(question_map[question_map_reverse[sid_str]]['content'])
                
                
        return questions
                
    except Exception as e:
        logger.error(f"获取问题详情失败: {str(e)}")
        return [f"Error: 无法获取问题ID {sid} 的详情" for sid in sids]


def get_correct(student_id: int, question_id: int, dataset_name: str) -> int:
    """
    获取学生对特定问题的回答是否正确
    
    Args:
        student_id: 学生ID
        question_id: 问题ID
        dataset_name: 数据集名称
        
    Returns:
        1表示正确，0表示错误，-1表示找不到数据
    """
    path = os.path.join('data', dataset_name, 'test_triples.csv')
    
    try:
        df = pd.read_csv(path)
        df.set_index(['student_id', 'question_id'], inplace=True)
        
        try:
            # 获取正确性数据
            return df.loc[(student_id, question_id), 'correct'].item()
        except KeyError:
            logger.warning(f"找不到学生{student_id}对问题{question_id}的答案数据")
            return -1
            
    except Exception as e:
        logger.error(f"获取正确性数据失败: {str(e)}")
        return -1


@lru_cache(maxsize=128)
def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    获取文本的嵌入向量
    
    Args:
        text: 输入文本
        model: 使用的嵌入模型
        
    Returns:
        嵌入向量
    """
    try:
        # 预处理文本
        text = text.replace("\n", " ")
        
        # 创建OpenAI客户端
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        )
        
        # 获取嵌入
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
        
    except Exception as e:
        logger.error(f"获取嵌入失败: {str(e)}")
        # 返回全零向量作为后备
        return [0.0] * 1536  # ada-002嵌入维度为1536


def vector2description(vector: List[float], knowledge_map: Dict, id2name_map: Dict) -> Dict[str, float]:
    """
    将向量转换为描述性字典
    
    Args:
        vector: 输入向量
        knowledge_map: 知识点映射
        id2name_map: ID到名称的映射
        
    Returns:
        每个知识点及其相应值的字典
    """
    vector2des = {}
    
    for num, value in enumerate(vector):
        knowledge_id = knowledge_map.get(str(num))
        if knowledge_id:
            knowledge_name = id2name_map.get(str(knowledge_id), f"未知知识点_{num}")
            vector2des[knowledge_name] = float(value)
    
    return vector2des


def batch_process_queries(
    queries: List[str], 
    model: Any, 
    vecdb: Any, 
    questions: List[str], 
    idToidx: Dict[int, int]
) -> List[Tuple[int, str, float]]:
    """
    批量处理查询并返回最相似的结果
    
    Args:
        queries: 查询列表
        model: 嵌入模型
        vecdb: 向量数据库
        questions: 问题列表
        idToidx: ID到索引的映射
        
    Returns:
        每个查询的最相似结果
    """
    # 获取查询向量
    qvecs = np.asarray(model.encode(queries))
    faiss.normalize_L2(qvecs)
    
    # 搜索最相似项
    sims, idcs = vecdb.search(qvecs, 1)
    
    results = []
    for idx, query_vec in enumerate(qvecs):
        most_similar_idx = idcs[idx][0]
        most_similar_question = questions[most_similar_idx]
        similarity_score = sims[idx][0]
        original_id = idToidx[most_similar_idx]
        results.append((original_id, most_similar_question, similarity_score))
    
    return results