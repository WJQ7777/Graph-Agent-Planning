import pandas as pd
import json

def add_index_to_extra_info(parquet_file_path, output_file_path=None):
    """
    给 parquet 文件中的 extra_info 字段添加 index 信息
    
    Args:
        parquet_file_path: 输入的 parquet 文件路径
        output_file_path: 输出文件路径，如果为 None 则覆盖原文件
    """
    # 读取 parquet 文件
    df = pd.read_parquet(parquet_file_path)
    
    print(f"原始数据集大小: {len(df)} 行")
    
    # 检查是否存在 extra_info 字段
    if 'extra_info' not in df.columns:
        print("错误: 数据集中没有找到 extra_info 字段")
        return
    
    # 处理每一行的 extra_info
    def add_index_to_row(row_index, extra_info_value):
        try:
            # 如果 extra_info 是字符串，尝试解析为 JSON
            if isinstance(extra_info_value, str):
                extra_info_dict = json.loads(extra_info_value)
            elif isinstance(extra_info_value, dict):
                extra_info_dict = extra_info_value.copy()
            else:
                # 如果是其他类型，创建新的字典
                extra_info_dict = {}
            
            # 添加 index 字段
            extra_info_dict['index'] = row_index
            
            return extra_info_dict
            
        except (json.JSONDecodeError, TypeError) as e:
            print(f"警告: 第 {row_index} 行的 extra_info 解析失败: {e}")
            # 创建新的字典包含 index
            return {'index': row_index}
    
    # 应用处理函数
    df['extra_info'] = df.reset_index().apply(
        lambda row: add_index_to_row(row.name, row['extra_info']), 
        axis=1
    )
    
    # 确定输出路径
    if output_file_path is None:
        output_file_path = parquet_file_path
    
    # 保存处理后的数据
    df.to_parquet(output_file_path, index=False)
    
    print(f"处理完成! 数据已保存到: {output_file_path}")
    print(f"处理后数据集大小: {len(df)} 行")
    
    # 验证结果 - 显示前几行的 extra_info
    print("\n前3行的 extra_info 示例:")
    for i in range(min(3, len(df))):
        print(f"行 {i}: {df.iloc[i]['extra_info']}")

def verify_index_loading(parquet_file_path, num_samples=5):
    """
    验证 index 能否正确加载
    
    Args:
        parquet_file_path: parquet 文件路径
        num_samples: 验证的样本数量
    """
    df = pd.read_parquet(parquet_file_path)
    
    print(f"\n验证 index 加载 (前 {num_samples} 行):")
    for i in range(min(num_samples, len(df))):
        row_dict = df.iloc[i].to_dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        print(f"行 {i}: 加载的 index = {index}")

if __name__ == "__main__":
    # 使用示例
    input_file = "/home/jiaqi/Agent_Foundation_Models/AFM/data/mhqa_agent/AFM-MHQA-RL-Dataset/original_16w_column_filter.parquet"
    output_file = "/home/jiaqi/Agent_Foundation_Models/AFM/data/mhqa_agent/AFM-MHQA-RL-Dataset/original_16w_column_filter_index.parquet"  # 可以设置为 None 来覆盖原文件
    
    # 添加 index 信息
    add_index_to_extra_info(input_file, output_file)
    
    # 验证结果
    verify_index_loading(output_file if output_file else input_file)