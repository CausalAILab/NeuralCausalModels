import numpy as np
import pandas as pd
import torch as T

from .base_pipeline import BasePipeline
from src import metric
from src.ds import CausalGraph
from src.scm import NCM


class NLLNCMMaxPipeline(BasePipeline):
    patience = 200
    biased = False

    def __init__(self, ctm, dat, cg_file, maximize=True, max_reg_upper=0.1, max_reg_lower=0.001, total_iters=1000):
        if isinstance(cg_file, str):
            parsed_cg = CausalGraph.read(cg_file)
        elif isinstance(cg_file, CausalGraph):
            parsed_cg = cg_file
        else:
            raise Exception("Unrecognized causal diagram data format.")
        ncm = NCM(parsed_cg)
        super().__init__(ctm, dat, parsed_cg, ncm)

        self.automatic_optimization = False
        self.nlpv = metric.probability_table(dat=dat)
        self.nlpv['_nlpv'] = -np.log(self.nlpv['P(V)'].astype(np.float32))

        self.maximize = maximize
        self.max_reg_upper = max_reg_upper
        self.max_reg_lower = max_reg_lower
        self.total_iters = total_iters
        self.max_reg = 1.0

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
            'lr_scheduler': T.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optim, 20, 1, eta_min=1e-4)
        }

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        n = int(2 * 10 ** 4)
        reg_ratio = min(self.current_epoch, self.total_iters) / self.total_iters
        reg_up = np.log(self.max_reg_upper)
        reg_low = np.log(self.max_reg_lower)
        max_reg = np.exp(reg_up - reg_ratio * (reg_up - reg_low))

        loss_agg = 0
        nll_agg = []
        nlpv_agg = []
        space = list(self.ncm.space())
        opt.zero_grad()
        for v, nlpv in zip(space, self.nlpvs):
            # _, nll = self.ncm.nll(v, m=n, return_biased=self.biased)
            nll = self.ncm.biased_nll(v, n=n)
            loss = T.exp(-nlpv) * (nll - nlpv)
            self.manual_backward(loss, opt)
            nll_agg.append(nll.item())
            nlpv_agg.append(nlpv.item())
            loss_agg += loss.item()
            del nll, loss
        # print("\nNLL Loss: {}".format(loss_agg))
        max_loss = max_reg * self.ate_loss(n)
        # print("Max Reg: {}".format(max_reg))
        # print("Max Loss: {}".format(max_loss.item()))
        self.manual_backward(max_loss, opt)
        loss_agg += max_loss.item()
        del max_loss
        # T.nn.utils.clip_grad_norm_(self.ncm.parameters(), 1e-7)
        opt.step()
        self.log('train_loss', loss_agg, prog_bar=True)
        self.log('lr', opt.param_groups[0]['lr'], prog_bar=True)

        # logging
        if (self.current_epoch + 1) % 10 == 0:
            results = metric.all_metrics(self.ctm, self.ncm, self.dat,
                                         self.cg_file, n=n)
            for k, v in results.items():
                if self.maximize:
                    self.log("max_" + k, v)
                else:
                    self.log("min_" + k, v)

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
            print(pd.Series(results))

    def precision_check(self, val):
        return T.relu(val) + 0.000001

    def ate_loss(self, n=1000000):
        y0_do0 = T.exp(-self.ncm.nll_marg({'Y': T.LongTensor([[0]]).to(self.device)}, m=n,
                                         do={'X': T.LongTensor([[0]]).to(self.device)}, return_biased=self.biased))
        y0_do1 = T.exp(-self.ncm.nll_marg({'Y': T.LongTensor([[0]]).to(self.device)}, m=n,
                                         do={'X': T.LongTensor([[1]]).to(self.device)}, return_biased=self.biased))
        y1_do0 = T.exp(-self.ncm.nll_marg({'Y': T.LongTensor([[1]]).to(self.device)}, m=n,
                                         do={'X': T.LongTensor([[0]]).to(self.device)}, return_biased=self.biased))
        y1_do1 = T.exp(-self.ncm.nll_marg({'Y': T.LongTensor([[1]]).to(self.device)}, m=n,
                                         do={'X': T.LongTensor([[1]]).to(self.device)}, return_biased=self.biased))

        if self.maximize:
            #return (-ate + 1.0) / 2.0
            y1_do1_norm = (y1_do1 / (y0_do1 + y1_do1)).float()
            y0_do0_norm = (y0_do0 / (y0_do0 + y1_do0)).float()
            if (y0_do0_norm <= 0).any():
                y0_do0_norm = self.precision_check(y0_do0_norm)
            if (y1_do1_norm <= 0).any():
                y1_do1_norm = self.precision_check(y1_do1_norm)
            ate_pen = -T.log(y1_do1_norm.float()).mean() - T.log(y0_do0_norm.float()).mean()
            return ate_pen
        else:
            #return (ate + 1.0) / 2.0
            y1_do0_norm = (y1_do0 / (y0_do0 + y1_do0)).float()
            y0_do1_norm = (y0_do1 / (y0_do1 + y1_do1)).float()
            if (y1_do0_norm <= 0).any():
                y1_do0_norm = self.precision_check(y1_do0_norm)
            if (y0_do1_norm <= 0).any():
                y0_do1_norm = self.precision_check(y0_do1_norm)
            ate_pen = -T.log(y0_do1_norm.float()).mean() - T.log(y1_do0_norm.float()).mean()
            return ate_pen

    def tv_loss(self, n=1000000):
        y0x0 = T.exp(-self.ncm.nll_marg({'Y': T.LongTensor([[0]]).to(self.device),
                                         'X': T.LongTensor([[0]]).to(self.device)},
                                        m=n, return_biased=self.biased))
        y0x1 = T.exp(-self.ncm.nll_marg({'Y': T.LongTensor([[0]]).to(self.device),
                                         'X': T.LongTensor([[1]]).to(self.device)},
                                        m=n, return_biased=self.biased))
        y1x0 = T.exp(-self.ncm.nll_marg({'Y': T.LongTensor([[1]]).to(self.device),
                                         'X': T.LongTensor([[0]]).to(self.device)},
                                        m=n, return_biased=self.biased))
        y1x1 = T.exp(-self.ncm.nll_marg({'Y': T.LongTensor([[1]]).to(self.device),
                                         'X': T.LongTensor([[1]]).to(self.device)},
                                        m=n, return_biased=self.biased))

        y1_g0 = y1x0 / (y1x0 + y0x0)
        y1_g1 = y1x1 / (y1x1 + y0x1)
        tv = y1_g1.float().mean() - y1_g0.float().mean()

        return tv
