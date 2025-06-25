import os
import argparse
from pathlib import Path
from typing import Union

import zhconv
from tqdm import tqdm

from new.text_normalize import NSWNormalizer

def traditional_to_simplified(text):
    simplified_text = zhconv.convert(text, 'zh-hans')
    chinese_text = ''.join(char for char in simplified_text if '\u4e00' <= char <= '\u9fff')
    chinese_text = NSWNormalizer(simplified_text).normalize(remove_punc=False).lower()
    return chinese_text

def srt_to_text(path: Union[str, Path]) -> str:
    """从SRT文件中提取文本内容"""
    text = ""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # 跳过时间戳和序号行
            if '-->' not in line and not line.strip().isdigit() and line.strip():
                text += line.strip() + " "
    return text.strip()

def split_chinese_chars(text):
    """将中文文本按字符分开"""
    return ' '.join(list(text))

def main():
    parser = argparse.ArgumentParser(description="评估转录结果并显示对比")
    parser.add_argument(
        "--recognized-dir",
        type=str,
        required=True,
        help="包含模型转录结果的SRT文件目录"
    )
    parser.add_argument(
        "--transcript-dir",
        type=str,
        required=True,
        help="包含参考文本的SRT文件目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录，用于保存评估结果和对比文件"
    )
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    temp_text = os.path.join(args.output_dir, 'temp_text')
    temp_ground_truth = os.path.join(args.output_dir, 'temp_ground_truth')
    results = os.path.join(args.output_dir, 'cer')
    comparison = os.path.join(args.output_dir, 'comparison.txt')

    # 准备对比文件
    with open(temp_text, 'w', encoding='utf-8') as f_pred, \
         open(temp_ground_truth, 'w', encoding='utf-8') as f_truth, \
         open(comparison, 'w', encoding='utf-8') as f_comp:
        
        print("处理转录文件...")
        for recognized_path in tqdm(list(Path(args.recognized_dir).iterdir())):
            speech_id = recognized_path.stem
            transcript_path = Path(args.transcript_dir) / f"{speech_id}.srt"
            
            if not transcript_path.exists():
                print(f"警告：找不到对应的参考文件：{transcript_path}")
                continue

            # 读取并标准化文本
            rec_text = traditional_to_simplified(srt_to_text(recognized_path))
            ref_text = traditional_to_simplified(srt_to_text(transcript_path))

            # 写入标准化后的文本用于计算CER（添加ID前缀并分字）
            f_pred.write(f"{speech_id} {split_chinese_chars(rec_text)}\n")
            f_truth.write(f"{speech_id} {split_chinese_chars(ref_text)}\n")

            # 写入对比信息（原始格式，方便人工查看）
            f_comp.write(f"音频: {speech_id}\n")
            f_comp.write(f"参考: {ref_text}\n")
            f_comp.write(f"识别: {rec_text}\n")
            f_comp.write("-" * 80 + "\n")

    # 计算CER
    print("\n计算字符错误率(CER)...")
    os.system(f'python new/compute-wer.py --char=1 --v=1 {temp_ground_truth} {temp_text} > {results}')

    # 显示结果
    print("\n评估完成！结果文件：")
    print(f"1. 详细对比：{comparison}")
    print(f"2. CER结果：{results}")

if __name__ == "__main__":
    main() 