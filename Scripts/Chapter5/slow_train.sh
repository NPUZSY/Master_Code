# 从零开始
nohup python /home/siyu/Master_Code/Scripts/Chapter5/slow_training.py \
--num-epochs 1000 \
--gamma 0.99 \
--lr 0.0001 \
--load-model-path /home/siyu/Master_Code/nets/Chap5/slow_training/0105_113601/slow_training_model_best.pth \
> logs/0105/0105_2.log 2>&1 &