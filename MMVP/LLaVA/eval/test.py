import json

def extract_responses(input_file, output_file):
    responses = []
    
    # 读取JSONL文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 解析每行JSON
                entry = json.loads(line.strip())
                if 'response' in entry:
                    responses.append(entry['response'])
            except json.JSONDecodeError:
                print(f"警告：跳过无效JSON行 -> {line}")
                continue
    
    # 写入输出JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_responses": len(responses),
            "responses": responses
        }, f, indent=2, ensure_ascii=False)
    
    print(f"成功提取 {len(responses)} 个response到 {output_file}")

if __name__ == "__main__":
    # 配置输入输出路径
    input_jsonl = "./answer1.jsonl"    # 输入文件路径
    output_json = "output1.json"    # 输出文件路径
    
    extract_responses(input_jsonl, output_json)