import numpy as np
import pandas as pd
import torch as T

from src import metric
from src.scm import CTM

from .base_pipeline import BasePipeline


class NLLCTMPipeline(BasePipeline):
    patience = 30

    def __init__(self, ctm, dat, cg_file):
        ncm = CTM(cg_file)
        super().__init__(ctm, dat, cg_file, ncm)

        self.automatic_optimization = False
        self.nlpv = metric.probability_table(dat=dat)
        self.nlpv['_nlpv'] = -np.log(self.nlpv['P(V)'].astype(np.float32))

    def forward(self, n=1000, u=None, do={}):
        return self.ncm(n, u, do)

    def configure_optimizers(self):
        optim = T.optim.SGD(self.ncm.parameters(), lr=1,
                            momentum=.9, nesterov=True)
        return {
            'optimizer': optim,
            #             'lr_scheduler': T.optim.lr_scheduler.ReduceLROnPlateau(
            #                 optim, patience=10, factor=.5),
            #             'monitor': 'train_loss'
        }

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        loss_agg = 0
        nll_agg = []
        nlpv_agg = []
        space = list(self.ncm.space())
        nlpvs = [T.tensor(self.nlpv.query(
            ' and '.join(f'{k} == {val.item()}'
                         for k, val in v.items()))._nlpv.item())
                 for v in space]
        opt.zero_grad()
        for v, nlpv in zip(space, nlpvs):
            nll = -self.ncm.log_pmf(v)
            loss = T.exp(-nlpv) * (nll - nlpv)
            self.manual_backward(loss, opt)
            nll_agg.append(nll.item())
            nlpv_agg.append(nlpv.item())
            loss_agg += loss.item()
            del nll, loss
        opt.step()
        self.log('train_loss', loss_agg, prog_bar=True)
        self.log('lr', opt.param_groups[0]['lr'], prog_bar=True)

        # logging
        if (self.current_epoch + 1) % 50 == 0:
            with T.no_grad():
                nlpv = T.tensor(nlpv_agg)
                nll = T.tensor(nll_agg)
                arr = T.stack([
                    T.exp(-nlpv),
                    T.exp(-nll),
                    T.exp(-nlpv) * (nll - nlpv)
                ], dim=-1).numpy()
                print(pd.DataFrame(arr, columns=['P*(V)', 'P(V)', 'loss']))
            results = metric.all_metrics(self.ctm, self.ncm, self.dat,
                                         self.cg_file, n=10000)
            print(pd.Series(results))
            for k, v in results.items():
                self.log(k, v)
