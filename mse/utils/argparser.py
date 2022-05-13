import argparse

__all__ = ['args']

parser = argparse.ArgumentParser(description='Process arguments')

# environment arguments
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--sinkhorn_device', type=int, default=2)

# saving arguments
parser.add_argument('--save_dir', type=str, default='checkpoints')
parser.add_argument('-sc', '--save_checkpoint', action='store_true')

# data arguments
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('-f','--flatten', action='store_true')

# training arguments
parser.add_argument('-lop', '--label_optimize_period', type=int, default=1)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0005)

parser.add_argument('--warmup_epoch', type=int, default=10)

# sinkhorn arguments
parser.add_argument('--num_clusters', type=int, default=4)
parser.add_argument('--num_head', type=int, default=1)
parser.add_argument('--lamb', type=float, default=25)
parser.add_argument('--dist', type=str, default='gauss', choices=['gauss', 'uniform'])
parser.add_argument('--gauss_sd', type=float, default=0.1)

# eval arguments
parser.add_argument('-hi', '--hierarchical', action='store_true')
parser.add_argument('-k', '--kmeans', action='store_true')
parser.add_argument('-r', '--reducer', type=str, choices=['pca', 'tsne', 'umap'], default='pca')
parser.add_argument('-n', '--eval_num_clusters', type=int, default=4)

args = parser.parse_args()