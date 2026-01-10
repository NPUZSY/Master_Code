# 从零开始
nohup python /home/siyu/Master_Code/Scripts/Chapter5/slow_training.py \
--num-epochs 5000 \
--epsilon 0.2 \
--gamma 0.95 \
--lr 0.01 \
--hidden-dim 512 \
--pool-size 200 \
> logs/0110/slow_train_0.log 2>&1 &

# --load-model-path /home/siyu/Master_Code/nets/Chap5/slow_training/0109_111647/slow_training_model_best.pth \







# 从joint_net开始
# nohup python /home/siyu/Master_Code/Scripts/Chapter5/slow_training.py \
# --epsilon 0.2 \
# --from-joint-net /home/siyu/Master_Code/nets/Chap4/Joint_Net/1223/2 \
# --num-epochs 1000 \
# --hidden-dim 256 \
# --pool-size 200 \
# > logs/0109/slow_train_from_joint.log 2>&1 &