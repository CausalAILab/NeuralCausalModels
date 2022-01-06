import numpy as np
import pandas as pd
import torch as T

from src import metric
from src.ds import CausalGraph
from src.scm import NCM

from .base_pipeline import BasePipeline


class NLLNCMPipeline(BasePipeline):
    patience = 60

    def __init__(self, ctm, dat, cg_file):
        ncm = NCM(CausalGraph.read(cg_file))
        super().__init__(ctm, dat, cg_file, ncm)

        self.automatic_optimization = False
        self.nlpv = metric.probability_table(dat=dat)
        self.nlpv['_nlpv'] = -np.log(self.nlpv['P(V)'].astype(np.float32))

        self.accumulate_batches = 1
        self.last_loss = None
        self.last_loss_sem = None

        space = list(self.ncm.space())
        self.nlpvs = [T.tensor(
            (lambda x: 0 if len(x) == 0 else x.item())
            (self.nlpv.query(' and '.join(f'{k} == {val.item()}'
                                          for k, val in v.items()))._nlpv)).float()
            for v in space]

    def forward(self, n=1000, u=None, do={}):
        return self.ncm(n, u, do)

    def configure_optimizers(self):
        optim = T.optim.AdamW(self.ncm.parameters(), lr=4e-3)
        return {
            'optimizer': optim,
            'lr_scheduler': T.optim.lr_scheduler.ReduceLROnPlateau(optim),
            'monitor': 'train_loss'
        }

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 25:
            self.accumulate_batches = 2
        if self.last_loss is not None and self.last_loss_sem > T.abs(self.last_loss) * 0.25:
            self.accumulate_batches = min(self.accumulate_batches * 2, 32)

        m = int(1.6e6)
        opt = self.optimizers()
        nll_agg = []
        nlpv_agg = []
        opt.zero_grad()
        space = list(self.ncm.space())
        biased_losses = []
        losses = []
        for _ in range(self.accumulate_batches):
            batch_loss = 0
            batch_biased_loss = 0
            for v, nlpv in zip(space, self.nlpvs):
                nll, biased_nll = self.ncm.nll(v, m=m, return_biased=True)
                loss = T.exp(-nlpv) * (nll - nlpv)
                biased_loss = T.exp(-nlpv) * (biased_nll - nlpv)
                self.manual_backward(loss, opt)
                nll_agg.append(nll.item())
                nlpv_agg.append(nlpv.item())
                batch_loss += loss.item()
                batch_biased_loss += biased_loss.item()
                del nll, loss
            losses.append(batch_loss)
            biased_losses.append(batch_biased_loss)
        opt.step()
        losses = T.tensor(losses)
        self.last_loss = losses.mean()
        self.last_loss_sem = losses.std() / (self.accumulate_batches ** .5)
        self.log('train_loss', self.last_loss, prog_bar=True)
        self.log('biased_loss', T.tensor(biased_losses).mean(), prog_bar=True)
        self.log('sem', self.last_loss_sem, prog_bar=True)
        self.log('lr', opt.param_groups[0]['lr'], prog_bar=True)
        self.log('acc', self.accumulate_batches, prog_bar=True)

        # logging
        if (self.current_epoch + 1) % 10 == 0:
            results = metric.all_metrics(self.ctm, self.ncm, self.dat,
                                         self.cg_file, n=10000)
            for k, v in results.items():
                self.log(k, v)

        if (self.current_epoch + 1) % 200 == 0:
            with T.no_grad():
                nlpv = T.tensor(nlpv_agg)
                nll = T.tensor(nll_agg)
                arr = T.stack([
                    T.exp(-nlpv),
                    T.exp(-nll),
                    T.exp(-nlpv) * (nll - nlpv)
                ], dim=-1).numpy()
                print(pd.DataFrame(arr, columns=['P*(V)', 'P(V)', 'loss']))
            print(pd.Series(results))
