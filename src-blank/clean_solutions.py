#!/usr/bin/env python3
"""
clean_solutions.py
一键清理 TinyTorch src/ 目录下所有模块的 SOLUTION 答案代码

功能：
- 递归处理 src/ 下所有子文件夹的 .py 文件
- 只删除 ### BEGIN SOLUTION 和 ### END SOLUTION 之间的实现代码
- 保留两行标记，并在中间插入一个空白行（让你知道哪里要写代码）
- 自动备份原文件为 .bak（安全第一）

使用方法：
1. 把此脚本保存到 src/clean_solutions.py
2. cd 到 src/ 目录
3. python clean_solutions.py
"""

import os
from pathlib import Path

def clean_solution_blocks(file_path: str):
    """清理单个 .py 文件中的 SOLUTION 块"""
    file_path = Path(file_path)
    backup_path = file_path.with_suffix('.py.bak')
    
    # 备份原文件
    if not backup_path.exists():
        file_path.rename(backup_path)
        print(f"✅ 已备份: {backup_path.name}")
    else:
        print(f"⚠️  已存在备份: {backup_path.name}（跳过备份）")
    
    # 读取文件
    with open(backup_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        if stripped == "### BEGIN SOLUTION":
            new_lines.append(line)                    # 保留 BEGIN 行
            new_lines.append("\n")                    # 插入一个空白行
            # 跳过 BEGIN 和 END 之间的所有内容
            i += 1
            while i < len(lines):
                if lines[i].strip() == "### END SOLUTION":
                    new_lines.append(lines[i])        # 保留 END 行
                    break
                i += 1
        else:
            new_lines.append(line)
        
        i += 1
    
    # 写回清理后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✅ 已清理: {file_path.name}")

def main():
    src_dir = Path(__file__).parent  # 当前脚本所在目录（src/）
    print("🚀 开始清理 TinyTorch src/ 目录下的所有 SOLUTION 代码...\n")
    
    py_files = list(src_dir.rglob("*.py"))
    
    # 排除本脚本自身
    py_files = [f for f in py_files if f.name != "clean_solutions.py"]
    
    if not py_files:
        print("❌ 未找到任何 .py 文件")
        return
    
    for py_file in py_files:
        # 只处理模块文件夹里的 .py（01_tensor、02_activations 等）
        if any(part.startswith(('0', '1', '2')) and '_' in part for part in py_file.parts):
            clean_solution_blocks(py_file)
    
    print("\n🎉 全部清理完成！")
    print("   - 原文件已备份为 .bak")
    print("   - 现在你可以开始实现代码了")
    print("   - 运行 tito module start XX 即可看到干净的作业版 Notebook")

if __name__ == "__main__":
    main()