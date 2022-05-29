## Test Guideline
- Dataset
  - Check whether the dataset is in the correct format(dimension, random shuffle, label rest, indices, etc.)
- Model
  - Conv1D
  - Fully connected
  - LSTM
  - Transformer
- Trainer
  - FullSupervisedTrainer *Test whether the Model and dataset can work well*
  - HybridTrainer
- LabelOptimizer
  - SinkhornLabelOptimizer
- Evaluator
  - Eval


## Experiment Combination
1. (NonTemporal, Contasive)  + Fully Connected + Hybrid Trainer + Eval(KMeans, umap) 
    > F1: 0.005, Acc: 0.03
2. (NonTemporal, Contasive) + Fully Connected + FullSupervied Trainer + Pseudo Optimizer + Eval(KMeans, umap)
    > F1: 0.32, Acc: 0.56

3. (Temporal) + Conv1D + Sinkhorn Trainer + Eval(KMeans, umap)
    > Awaiting

4. (Temporal, Contasive) + Conv1D + Hybrid Trainer + Eval(KMeans, umap)
    > Awaiting

5. (Temporal, Contasive) + Conv1D + FullSupervied Trainer + Eval(KMeans, umap)
    > Awaiting