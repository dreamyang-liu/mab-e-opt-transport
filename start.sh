
device=$1
sinkhorn_device=$2
data_dir='./data/train.npy'


python main.py --config config.json \
--seed "model_name" \
-f \
--device $device \
--sinkhorn_device $sinkhorn_device \
--data_dir $data_dir \