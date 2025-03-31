import json
import sys

def compare_jsonl(file1_path, file2_path, output_path):
    """
    比较两个JSONL文件的response字段差异
    :param file1_path: 第一个JSONL文件路径
    :param file2_path: 第二个JSONL文件路径
    :param output_path: 差异结果输出路径
    """
    differences = []
    
    # 同步逐行读取两个文件
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        line_num = 0
        while True:
            line1 = f1.readline()
            line2 = f2.readline()
            
            # 同时到达文件末尾
            if not line1 and not line2:
                break
                
            line_num += 1  # 行号从1开始
            
            # 处理行数不一致的情况
            if not line1 or not line2:
                raise ValueError(f"文件行数不一致：文件1有{'更多' if line2 else '更少'}的行数 (行号:{line_num})")

            # 解析JSON行
            try:
                data1 = json.loads(line1.strip())
                data2 = json.loads(line2.strip())
            except json.JSONDecodeError as e:
                print(f"行号 {line_num} JSON解析错误: {str(e)}")
                continue
                
            # 提取response字段
            resp1 = data1.get('response', '<MISSING_FIELD>')
            resp2 = data2.get('response', '<MISSING_FIELD>')
            
            # 记录差异
            if resp1 != resp2:
                differences.append({
                    "line": line_num,
                    "first": resp1,
                    "second": resp2
                })

    # 写入差异文件
    with open(output_path, 'w') as f:
        json.dump(differences, f, indent=2, ensure_ascii=False)
        
    print(f"对比完成，共发现 {len(differences)} 处差异")

if __name__ == "__main__":
        
    file1 = "answer.jsonl"
    file2 = "answer1.jsonl"
    output = "dif-01.json"
    
    compare_jsonl(file1, file2, output)