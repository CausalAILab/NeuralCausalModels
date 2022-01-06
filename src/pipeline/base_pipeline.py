import pytorch_lightning as pl
import torch as T


class BasePipeline(pl.LightningModule):
    min_delta = 1e-6
    patience = 20
    max_epochs = 10000

    def __init__(self, ctm, dat, cg_file, ncm):
        super().__init__()
        self.ctm = ctm
        self.dat = dat
        self.cg_file = cg_file
        self.ncm = ncm

    def forward(self, n=1, u=None, do={}):
        return self.ncm(n, u, do)

    def train_dataloader(self):  # 1 epoch = 1 step
        return T.utils.data.DataLoader(
            T.utils.data.TensorDataset(T.zeros(1, 1)),
            batch_size=1)
