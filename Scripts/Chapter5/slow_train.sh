# 从零开始
nohup python /home/siyu/Master_Code/Scripts/Chapter5/slow_training.py \
--num-epochs 1000 \
--gamma 0.99 \
--lr 0.0001 \
--hidden-dim 256 \
> logs/0105/0105_3.log 2>&1 &
# --load-model-path /home/siyu/Master_Code/nets/Chap5/slow_training/0105_113601/slow_training_model_best.pth \



# 从joint_net开始
nohup python Scripts/Chapter5/slow_training.py \
--num-epochs 5 \
--from-joint-net /home/siyu/Master_Code/nets/Chap4/Joint_Net/1223/2 \
--num-epochs 1000 \
--hidden-dim 256 \
> logs/0103/0105_4.log 2>&1 &