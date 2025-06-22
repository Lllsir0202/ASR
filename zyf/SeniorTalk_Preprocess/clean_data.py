# 把该py文件的 ./transcript 文件夹下面的所有文件依次取出来，
# 看看是否包含（）这个符号，付过包含，打印出来，并且把（）和他里面的内容删除，把处理好的文本放回原文件之中

import os
import re

def clean_line(line):
    result = []  # 用于存储清洗后的字符
    in_brackets = False  # 标记是否在括号内

    for char in line:
        if char == '（':
            # 遇到左括号，标记为在括号内
            in_brackets = True
        elif char == '）':
            # 遇到右括号，标记为不在括号内
            in_brackets = False
        elif not in_brackets:
            # 如果不在括号内，保留字符
            result.append(char)

    # 将字符列表拼接成字符串并返回
    return ''.join(result)


def process_files(folder_path):
    """
    遍历指定文件夹中的所有文件，逐行清洗包含括号的文本，并将结果写回原文件。
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在！")
        return

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # 确保是文件而不是子文件夹
        if os.path.isfile(file_path):
            try:
                # 创建临时文件
                temp_file_path = file_path + ".tmp"
                with open(file_path, 'r', encoding='utf-8') as infile, open(temp_file_path, 'w', encoding='utf-8') as outfile:
                    # 逐行读取文件
                    for line in infile:
                        # 清洗当前行
                        cleaned_line = clean_line(line)
                        # 写入清洗后的行到临时文件
                        outfile.write(cleaned_line)

                # 替换原文件
                os.replace(temp_file_path, file_path)
                print(f"已清洗并保存到: {file_name}")

            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")

if __name__ == "__main__":
    # 指定文件夹路径
    transcript_folder = "./transcript"

    # 调用函数处理文件
    process_files(transcript_folder)