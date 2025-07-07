import csv


def process_text_file_to_csv(input_filename='input.txt', output_filename='output.csv'):
    """
    遍历文本文件，提取特定数据并写入CSV文件。

    :param input_filename: 输入的文本文件名。
    :param output_filename: 输出的CSV文件名。
    """
    # 指定CSV文件的列名，作为我们提取数据的“键”
    column_keys = ['1', '5', '11', '13', '15', '20']

    # 初始化一个字典来存储每一列的数据
    # 例如：{'1': [96, 88], '5': [105], ...}
    data = {key: [] for key in column_keys}

    try:
        # 打开并读取整个文本文件
        with open(input_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{input_filename}'。请确保文件存在且路径正确。")
        return
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}")
        return

    # 遍历文件的每一行（及其索引）
    for i, line in enumerate(lines):
        # 检查当前行是否以 ': 25000' 结尾
        if line.strip().endswith(': 25000'):
            # 计算目标行的索引（当前行往下第四行）
            target_index = i + 4

            # 确保目标行存在，防止文件末尾越界
            if target_index < len(lines):
                target_line = lines[target_index].strip()

                try:
                    # 尝试按冒号分割目标行
                    parts = target_line.split(':')
                    if len(parts) == 2:
                        # 提取冒号前后的数字
                        key = parts[0].strip()
                        value = int(parts[1].strip())

                        # 检查key是否是我们关注的列之一
                        if key in column_keys:
                            # 将提取到的value添加到对应key的列表中
                            data[key].append(value)

                except (ValueError, IndexError):
                    # 如果目标行格式不正确（例如，无法转换为整数），则打印警告并跳过
                    print(f"警告：无法解析第 {target_index + 1} 行的内容：'{target_line}'")

    # --- 数据写入CSV文件 ---

    # 找出所有数据列中最长的长度
    max_len = 0
    for key in data:
        if len(data[key]) > max_len:
            max_len = len(data[key])

    # 将所有较短的列表用空字符串填充，以保证所有列长度一致
    for key in data:
        while len(data[key]) < max_len:
            data[key].append('')

    try:
        # 创建并写入CSV文件
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 写入列名（表头）
            writer.writerow(column_keys)

            # 逐行写入数据
            # 循环次数为最长列表的长度，也就是CSV文件的行数
            for i in range(max_len):
                # 构建每一行的数据
                row = [data[key][i] for key in column_keys]
                writer.writerow(row)

        print(f"处理完成！数据已成功写入 '{output_filename}'。")

    except IOError:
        print(f"错误：无法写入CSV文件 '{output_filename}'。请检查文件权限。")
    except Exception as e:
        print(f"写入CSV时发生未知错误: {e}")


# --- 如何使用 ---
if __name__ == '__main__':
    # 假设您的文本文件名为 'input.txt'
    # 假设您希望生成的CSV文件名为 'output.csv'
    process_text_file_to_csv(r'C:\Users\30288\AppData\Local\Temp\Mxt251\RemoteFiles\4656612_4_7\UUUZQMODE.out', r'C:\Users\30288\OneDrive\Desktop\ctmd\UUUZQMODE1.csv')