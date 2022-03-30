# from KMeansTrainer import KMeansTrainer as kmeans_trainer
# from SinkhornTrainer import SinkhornTrainer as sinkhorn_trainer
from trainers.mab_e_trainer import Trainer as mabe_trainer
from trainers.sinkhorn_trainer import SinkhornTrainer as sinkhorn_trainer

class TrainerFactory:
    
    @staticmethod
    def create_trainer(type='mabe'):
        if type == 'mabe':
            return mabe_trainer
        elif type == 'sinkhorn':
            return sinkhorn_trainer
