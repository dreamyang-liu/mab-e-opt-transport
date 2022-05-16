
seed=42
device=3
sinkhorn_device=4
save_dir='./save'
warmup_epoch=20
data_dir='./data/train.npy'
lop=2
epoch=50
batch_size=128
num_clusters=100
lr=1e-3
dist=gauss
gauss_sd=0.1
reducer=umap




python main.py \
--seed $seed \
-f \
-sc \
--device $device \
--sinkhorn_device $sinkhorn_device \
--warmup_epoch $warmup_epoch \
--data_dir $data_dir \
--batch_size $batch_size \
--lr $lr \
--save_dir $save_dir \
-lop $lop \
--epoch $epoch \
--num_clusters $num_clusters \
--dist $dist \
--gauss_sd $gauss_sd \
--reducer $reducer \