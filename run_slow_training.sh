#!/bin/bash

# 确保logs目录存在
mkdir -p logs

# 后台运行慢训练脚本，输出重定向到指定日志文件
nohup python Scripts/Chapter5/slow_training.py --num-epochs 5000 --load-model-path /home/siyu/Master_Code/nets/Chap5/slow_training/0101_200526/slow_training_model_best.pth > logs/0102_1.log 2>&1 &

# 显示命令执行信息
echo "慢训练脚本已在后台启动，输出日志: logs/0102_1.log"
echo "可以使用 'tail -f logs/0102_1.log' 查看实时输出"
echo "可以使用 'ps aux | grep slow_training.py' 查看进程状态"