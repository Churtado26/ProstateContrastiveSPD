import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning import miners
from pytorch_metric_learning import distances
from pytorch_metric_learning.distances import LpDistance     
   
class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, positives = None, negatives = None):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        rules = {
            'easy': miners.BatchEasyHardMiner.EASY,
            'semihard': miners.BatchEasyHardMiner.SEMIHARD,
            'hard': miners.BatchEasyHardMiner.HARD
        }
        if positives is None or negatives is None:
            self.miner_function = None
            print('No mining will be applied')
        else:
            self.miner_function = miners.BatchEasyHardMiner(
                pos_strategy = rules[positives],
                neg_strategy = rules[negatives]
            )

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        if self.miner_function is None:
            return losses.NTXentLoss(
                temperature = self.temperature
            )(feature_vectors, torch.squeeze(labels))
        else:

            mined_hard_embeddings = self.miner_function(feature_vectors, torch.squeeze(labels, dim=1))
            return losses.NTXentLoss(
                temperature=self.temperature,
            )(feature_vectors, torch.squeeze(labels, dim=1), mined_hard_embeddings )

#Triplet, usa lp distance
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.05, positives = None, negatives = None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        rules = {
            'easy': miners.BatchEasyHardMiner.EASY,
            'semihard': miners.BatchEasyHardMiner.SEMIHARD,
            'hard': miners.BatchEasyHardMiner.HARD
        }
        if positives is None or negatives is None:
            self.miner_function = None
            print('No mining will be applied')
        else:
            self.miner_function = miners.BatchEasyHardMiner(
                pos_strategy = rules[positives],
                neg_strategy = rules[negatives]
            )


    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        if self.miner_function is None:
            return losses.TripletMarginLoss(
                margin = self.margin,
                swap = False,
                smooth_loss = False, 
                triplets_per_anchor = 'all'
            )(feature_vectors, torch.squeeze(labels))
        else:
            # print(f"feature_vectors shape: {feature_vectors.shape}")
            # print(f"labels shape: {labels.shape}")
            # print(f"labels shape: {labels.view(-1).shape}")
            mined_hard_embeddings = self.miner_function(feature_vectors, labels.view(-1)) #torch.squeeze(labels) Lo quite por un error de dimensiones
            return losses.TripletMarginLoss(
                margin=self.margin,
                swap = False,
                smooth_loss = False, 
                triplets_per_anchor = 'all'
            )(feature_vectors, labels.view(-1), mined_hard_embeddings )# aca igual
        
        
        


class NTXentLossLp(torch.nn.Module):
    def __init__(self, temperature=0.07, positives=None, negatives=None, p=2):
        super(NTXentLossLp, self).__init__()
        self.temperature = temperature
        self.p = 2  # Valor de p para la distancia Lp
        rules = {
            'easy': miners.BatchEasyHardMiner.EASY,
            'semihard': miners.BatchEasyHardMiner.SEMIHARD,
            'hard': miners.BatchEasyHardMiner.HARD
        }
        if positives is None or negatives is None:
            self.miner_function = None
            print('No mining will be applied')
        else:
            self.miner_function = miners.BatchEasyHardMiner(
                pos_strategy=rules[positives],
                neg_strategy=rules[negatives]
            )

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        if self.miner_function is None:
            return losses.NTXentLoss(
                temperature=self.temperature
            )(feature_vectors, torch.squeeze(labels))
        else:
            mined_hard_embeddings = self.miner_function(feature_vectors, torch.squeeze(labels, dim=1))
            distance_func = LpDistance(p=self.p)
            return losses.NTXentLoss(
                temperature=self.temperature,
                distance=distance_func
            )(feature_vectors, torch.squeeze(labels, dim=1), mined_hard_embeddings)

 