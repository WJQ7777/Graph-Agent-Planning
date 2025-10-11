import pandas as pd
import argparse
import os

def keep_first_n_rows(input_file, output_file=None, n_rows=10):
    """
    读取parquet文件，保留前n条数据，导出为新文件
    
    Args:
        input_file (str): 输入的parquet文件路径
        output_file (str): 输出的parquet文件路径，默认为None
        n_rows (int): 保留的行数，默认为10
    """
    try:
        # 读取parquet文件
        print(f"正在读取文件: {input_file}")
        df = pd.read_parquet(input_file)
        
        print(f"原始数据形状: {df.shape}")
        
        # 保留前n行
        df_subset = df.head(n_rows)
        print(f"保留前{n_rows}行后的数据形状: {df_subset.shape}")
        
        # 生成输出文件名
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_first_{n_rows}.parquet"
        
        # 保存为新的parquet文件
        df_subset.to_parquet(output_file, index=False)
        print(f"已保存到: {output_file}")
        
        # 显示前几行数据预览
        print("\n数据预览:")
        print(df_subset.head())
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='保留parquet文件的前N行数据')
    parser.add_argument('input_file', help='输入的parquet文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('-n', '--rows', type=int, default=500, help='保留的行数 (默认: 10)')
    
    args = parser.parse_args()
    
    keep_first_n_rows(args.input_file, args.output, args.rows)

if __name__ == "__main__":
    # 如果直接运行脚本，可以修改这里的参数
    # keep_first_n_rows("your_file.parquet", "output_file.parquet", 10)
    
    # 或者使用命令行参数
    main()