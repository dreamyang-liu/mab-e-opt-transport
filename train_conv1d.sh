
seed=42
device=3
data_dir='./data/train.npy'
epoch=50
batch_size=128
lr=1e-3
save_dir='./save/train_conv1d'
past_frames=100
future_frames=100
frame_skip=1



python train_conv1d.py \
--seed $seed \
-f \
-sc \
--device $device \
--data_dir $data_dir \
--batch_size $batch_size \
--lr $lr \
--save_dir $save_dir \
--epoch $epoch \
--past_frames $past_frames \
--future_frames $future_frames \
--frame_skip $frame_skip