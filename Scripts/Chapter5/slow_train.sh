# 从零开始
nohup python /home/siyu/Master_Code/Scripts/Chapter5/slow_training.py \
--num-epochs 2000 \
--epsilon 0.3 \
--gamma 0.9 \
--lr 0.0005 \
--hidden-dim 256 \
--pool-size 50 \
--load-model-path /home/siyu/Master_Code/nets/Chap5/slow_training/0113_100818/slow_training_model_best.pth \
> logs/0114/4_W2_20_W2_01_lr_00005_From0113_100818.log 2>&1 &





# # 从joint_net开始
# nohup python /home/siyu/Master_Code/Scripts/Chapter5/slow_training.py \
# --epsilon 0.2 \
# --from-joint-net /home/siyu/Master_Code/nets/Chap4/Joint_Net/1223/3 \
# --num-epochs 5000 \
# --hidden-dim 256 \
# --pool-size 200 \
# > logs/0110/slow_train_from_joint_tracking_reward.log 2>&1 &