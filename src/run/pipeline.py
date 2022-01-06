import argparse
import functools
import glob
import hashlib
import itertools
import json
import os
import shutil
from contextlib import contextmanager
from email.message import EmailMessage
from smtplib import SMTP
from socket import gethostname
from tempfile import NamedTemporaryFile
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch as T
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src import metric, pipeline
from src.ds import CausalGraph
from src.scm import CTM
from src.scm.model_families import XORModel
from src.pipeline import NLLNCMMaxPipeline
from src.metric.werm import werm_ate
from src.metric.tv_nn import naive_nn_tv


class L1Dataset(Dataset):
    @staticmethod
    def collate_fn(dicts):
        return {k: T.stack([dicts[i][k] for i in range(len(dicts))]) for k in dicts[0]}

    def __init__(self, v):
        self.v = {k: v[k] for k in v}
        self.n = len(next(iter(v.values())))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: self.v[k][i] for k in self.v}


@contextmanager
def training(net):
    '''Temporarily switch to training mode.'''
    mode = net.training
    try:
        net.train()
        yield net
    finally:
        net.train(mode=mode)


def interventional_is_observational(g, x='X', y='Y'):
    '''
    Return whether or not the causal effect of @x on @y is the same as the observational effect.
    '''
    g = CausalGraph(V=g.v,
                    directed_edges=[e for e in g.de if e[0] != x],
                    bidirected_edges=g.be)

    @functools.lru_cache(maxsize=None)
    def is_ancestor(v1, v2, allow_self=False):
        if v1 == v2:
            return allow_self
        return any(map(functools.partial(is_ancestor, v2=v2, allow_self=True),
                       (v1 if type(v1) is tuple else g.ch[v1])))

    return not any(is_ancestor(k, x) and is_ancestor(k, y)
                   for k in itertools.chain(g.v, g.bi))


def datagen(cg_file=None, n=500000, dat=None, dim=1, min_ate_tv_diff=0.05, adjust_ate=True):
    assert (cg_file is None) != (dat is None)

    # create true data-generating model
    g = CausalGraph.read(cg_file)
    if adjust_ate:
        ctm = CTM(cg_file, v_size={v: 1 if v in ('X', 'Y') else dim for v in g}).eval()
    else:
        ctm = CTM(cg_file).eval()

    # if P(Y | X) != P(Y | do(X)), increase |ate - tv| to >= min_ate_tv_diff
    if adjust_ate:
        if not interventional_is_observational(g):
            optim = T.optim.SGD(ctm.parameters(), lr=1)
            ate_tv_diff = T.tensor(0.)
            max_steps = 50000
            step = 0
            with training(ctm):
                assert ctm.training
                for _ in tqdm(range(max_steps), leave=False):
                    if ate_tv_diff.item() >= min_ate_tv_diff:
                        break

                    optim.zero_grad()
                    ate = (ctm.pmf({'Y': 1}, do={'X': T.tensor([[1]])})
                           - ctm.pmf({'Y': 1}, do={'X': T.tensor([[0]])}))
                    assert not T.isnan(ate), f'ate is nan, step {step}'
                    tv = (ctm.pmf({'Y': 1}, cond={'X': 1})
                          - ctm.pmf({'Y': 1}, cond={'X': 0}))
                    assert not T.isnan(tv), f'tv is nan, step {step}'
                    ate_tv_diff = T.abs(ate - tv)
                    (-ate_tv_diff).backward()
                    optim.step()
                else:
                    raise ValueError(f'> {max_steps} steps optimizing CTM')

    # generate data
    if dat is None:
        dat = ctm(n)

    return ctm, dat


@contextmanager
def lock(file, lockinfo):  # attempt to acquire a file lock; yield whether or not lock was acquired
    os.makedirs(os.path.dirname(file), exist_ok=True)
    os.makedirs('tmp/', exist_ok=True)
    with NamedTemporaryFile(dir='tmp/') as tmp:
        try:
            os.link(tmp.name, file)
            acquired_lock = True
        except FileExistsError:
            acquired_lock = os.stat(tmp.name).st_nlink == 2
    if acquired_lock:
        with open(file, 'w') as fp:
            fp.write(lockinfo)
        try:
            yield True
        finally:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
    else:
        yield False


def get_key(cg_file, n, dim, trial_index):
    graph = cg_file.split('/')[-1].split('.')[0]
    return ('graph=%s-n_samples=%s-dim=%s-trial_index=%s'
            % (graph, n, dim, trial_index))


def run(pipeline, cg_file, n, dim, trial_index, gpu=None,
        lockinfo=os.environ.get('SLURM_JOB_ID', ''), minmax=False):
    key = get_key(cg_file, n, dim, trial_index)
    d = 'out/%s/%s' % (pipeline.__name__, key)  # name of the output directory

    with lock(f'{d}/lock', lockinfo) as acquired_lock:
        if not acquired_lock:
            print('[locked]', d)
            return

        try:
            # return if best.th is generated (i.e. training is already complete)
            if os.path.isfile(f'{d}/best.th'):
                print('[done]', d)
                return

            # since training is not complete, delete all directory files except for the lock
            print('[running]', d)
            for file in glob.glob(f'{d}/*'):
                if os.path.basename(file) != 'lock':
                    if os.path.isdir(file):
                        shutil.rmtree(file)
                    else:
                        try:
                            os.remove(file)
                        except FileNotFoundError:
                            pass

            # set random seed to a hash of the parameter settings for reproducibility
            seed = int(hashlib.sha512(key.encode()).hexdigest(), 16) & 0xffffffff
            T.manual_seed(seed)
            np.random.seed(seed)
            print('Key:', key)
            print('Seed:', seed)

            # generate data-generating model, data, and model
            print('Generating data')
            ctm, dat = datagen(cg_file, n=n, dim=dim)
            m = pipeline(ctm, dat, cg_file)

            # print info
            print('ctm')
            print(metric.probability_table(ctm))
            print('dat')
            print(metric.probability_table(dat=dat))
            print('ncm')
            print(metric.probability_table(m.ncm))
            print('all metrics (prior to training)')
            print(metric.all_metrics(m.ctm, m.ncm, m.dat, cg_file))

            # function for building pytorch-lightning trainer
            def create_trainer(gpu=None):
                checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f'{d}/checkpoints/')
                return pl.Trainer(
                    callbacks=[
                        checkpoint,
                        pl.callbacks.EarlyStopping(monitor='train_loss',
                                                   patience=pipeline.patience,
                                                   min_delta=pipeline.min_delta,
                                                   check_on_train_epoch_end=True),
                    ],
                    max_epochs=pipeline.max_epochs,
                    accumulate_grad_batches=1,
                    logger=pl.loggers.TensorBoardLogger(f'{d}/logs/'),
                    log_every_n_steps=10,
                    terminate_on_nan=True,
                    gpus=gpu
                ), checkpoint

            # train model
            if minmax:
                m_min = pipeline(ctm, dat, cg_file, maximize=False)
                m_max = pipeline(ctm, dat, cg_file, maximize=True)
                if gpu is None:
                    gpu = int(T.cuda.is_available())
                trainer_min, checkpoint_min = create_trainer(gpu)
                trainer_max, checkpoint_max = create_trainer(gpu)
                print("\nTraining min model...")
                trainer_min.fit(m_min)
                ckpt = T.load(checkpoint_min.best_model_path)
                m_min.load_state_dict(ckpt['state_dict'])
                print("\nTraining max model...")
                trainer_max.fit(m_max)
                ckpt = T.load(checkpoint_max.best_model_path)
                m_max.load_state_dict(ckpt['state_dict'])

                results = metric.all_metrics_minmax(
                    m_min.ctm, m_min.ncm, m_max.ncm, m_min.dat, m_min.cg_file, n=100000)
                print(results)

                # save results
                with open(f'{d}/results.json', 'w') as file:
                    json.dump(results, file)
                T.save(dat, f'{d}/dat.th')
                T.save(dict(), f'{d}/best.th')  # breadcrumb file
                T.save(m_min.state_dict(), f'{d}/best_min.th')
                T.save(m_max.state_dict(), f'{d}/best_max.th')

                return (m_min, m_max), results
            else:
                m = pipeline(ctm, dat, cg_file)
                if gpu is None:
                    gpu = int(T.cuda.is_available())
                trainer, checkpoint = create_trainer(gpu)
                trainer.fit(m)
                # load_from_checkpoint() doesn't work
                ckpt = T.load(checkpoint.best_model_path)
                m.load_state_dict(ckpt['state_dict'])
                results = metric.all_metrics(
                    m.ctm, m.ncm, m.dat, m.cg_file, n=100000)
                print(results)

                # save results
                with open(f'{d}/results.json', 'w') as file:
                    json.dump(results, file)
                T.save(dat, f'{d}/dat.th')
                T.save(m.state_dict(), f'{d}/best.th')

                return m, results
        except Exception:
            # move out/*/* to err/*/*/#
            e = d.replace("out/", "err/").rsplit('-', 1)[0]
            e_index = len(glob.glob(e + '/*'))
            e += '/%s' % e_index
            os.makedirs(e.rsplit('/', 1)[0], exist_ok=True)
            shutil.move(d, e)
            print(f'moved {d} to {e}')
            raise


def run_id(folder_name, graph, graph_args, n, dim, n_epochs, trial_index, num_reruns,
           lockinfo=os.environ.get('SLURM_JOB_ID', ''), gpu=None):
    pipeline = NLLNCMMaxPipeline
    preset_graph = isinstance(graph_args, str)
    if preset_graph:
        parameters = ('graph=%s-n_samples=%s-dim=%s-n_epochs=%s-trial_index=%s'
                      % (graph_args, n, dim, n_epochs, trial_index))
    else:
        parameters = ('n_samples=%s-dim=%s-n_epochs=%s-n_vars=%s-p_dir=%s-p_bidir=%s-enf_dir=%s-enf_bidir=%s-'
                      'trial_index=%s'
                      % (n, dim, n_epochs, graph_args["n_vars"], graph_args["p_dir"], graph_args["p_bidir"],
                         graph_args["enf_dir"], graph_args["enf_bidir"], trial_index))

    # name of the output directory
    d = 'out/IDExperiments/%s/%s' % (folder_name, parameters)

    with lock(f'{d}/lock', lockinfo) as acquired_lock:
        if not acquired_lock:
            print('[locked]', d)
            return

        try:
            # return if best.th is generated (i.e. training is already complete)
            if os.path.isfile(f'{d}/{num_reruns - 1}/best_max.th'):
                print('[done]', d)
                return

            # since training is not complete, delete all directory files except for the lock
            print('[running]', d)

            # set random seed to a hash of the parameter settings for reproducibility
            seed = int(hashlib.sha512(parameters.encode()
                                      ).hexdigest(), 16) & 0xffffffff
            T.manual_seed(seed)
            np.random.seed(seed)
            print('Parameters:', parameters)
            print('Seed:', seed)

            # save causal diagram
            if not preset_graph:
                graph.save(f'{d}/graph.cg')

            # generate data
            if preset_graph:
                if os.path.isfile(f'{d}/gen_model.th') and os.path.isfile(f'{d}/dat.th'):
                    dat = T.load(f'{d}/dat.th')
                    gen_model = CTM("dat/cg/{}.cg".format(graph_args)).eval()
                    gen_model.load_state_dict(T.load(f'{d}/gen_model.th'))
                else:
                    gen_model, dat = datagen("dat/cg/{}.cg".format(graph_args), n=n, dim=dim, adjust_ate=False)
                    T.save(gen_model.state_dict(), f'{d}/gen_model.th')
                    T.save(dat, f'{d}/dat.th')
            else:
                gen_model = XORModel(graph, dim=dim, p=0.2, seed=seed)
                if os.path.isfile(f'{d}/dat.th'):
                    dat = T.load(f'{d}/dat.th')
                else:
                    dat = gen_model(n)
                    T.save(dat, f'{d}/dat.th')

            # function for building pytorch-lightning trainer
            def create_trainer(rerun_trial, gpu=None):
                checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f'{d}/{rerun_trial}/checkpoints/')
                trainer = pl.Trainer(
                    callbacks=[
                        checkpoint
                    ],
                    max_epochs=n_epochs,
                    accumulate_grad_batches=1,
                    logger=pl.loggers.TensorBoardLogger(f'{d}/{rerun_trial}/logs/'),
                    log_every_n_steps=10,
                    terminate_on_nan=True,
                    gpus=gpu
                )

                return trainer, checkpoint

            for r in range(num_reruns):
                if not os.path.isfile(f'{d}/{r}/best_max.th'):
                    # remove all files
                    for file in glob.glob(f'{d}/{r}/*'):
                        if os.path.isdir(file):
                            shutil.rmtree(file)
                        else:
                            try:
                                os.remove(file)
                            except FileNotFoundError:
                                pass

                    # reset seed
                    new_params = "{}-run={}".format(parameters, r)
                    seed = int(hashlib.sha512(new_params.encode()).hexdigest(), 16) & 0xffffffff
                    T.manual_seed(seed)
                    np.random.seed(seed)
                    print("Run {} seed: {}".format(r, seed))

                    # train model
                    m_min = pipeline(gen_model, dat, graph, maximize=False, max_reg_upper=1.0, max_reg_lower=0.001,
                                     total_iters=n_epochs)
                    m_max = pipeline(gen_model, dat, graph, maximize=True, max_reg_upper=1.0, max_reg_lower=0.001,
                                     total_iters=n_epochs)
                    if gpu is None:
                        gpu = int(T.cuda.is_available())
                    trainer_min, min_checkpoint = create_trainer(r, gpu)
                    trainer_max, max_checkpoint = create_trainer(r, gpu)
                    print("\nTraining min model...")
                    trainer_min.fit(m_min)
                    ckpt = T.load(min_checkpoint.best_model_path)
                    m_min.load_state_dict(ckpt['state_dict'])
                    print("\nTraining max model...")
                    trainer_max.fit(m_max)
                    ckpt = T.load(max_checkpoint.best_model_path)
                    m_max.load_state_dict(ckpt['state_dict'])

                    results = metric.all_metrics_minmax(
                        m_min.ctm, m_min.ncm, m_max.ncm, m_min.dat, m_min.cg_file, n=100000)
                    print(results)

                    # save results
                    with open(f'{d}/{r}/results.json', 'w') as file:
                        json.dump(results, file)
                    T.save(m_min.state_dict(), f'{d}/{r}/best_min.th')
                    T.save(m_max.state_dict(), f'{d}/{r}/best_max.th')
                else:
                    print("Done with run {}.".format(r))

            T.save(dict(), f'{d}/best.th')  # breadcrumb file
            return True
        except Exception:
            # move out/*/* to err/*/*/#
            e = d.replace("out/", "err/").rsplit('-', 1)[0]
            e_index = len(glob.glob(e + '/*'))
            e += '/%s' % e_index
            os.makedirs(e.rsplit('/', 1)[0], exist_ok=True)
            shutil.move(d, e)
            print(f'moved {d} to {e}')
            raise


def run_other_estimators(d, experiment, dev, redo_list,
                         lockinfo=os.environ.get('SLURM_JOB_ID', '')):
    with lock(f'{d}/lock', lockinfo) as acquired_lock:
        if not acquired_lock:
            print('[locked]', d)
            return

        try:
            w_ate = None
            naive_ate = None
            naive_kl = None

            if os.path.isfile("{}/ate_results.json".format(d)):
                with open("{}/ate_results.json".format(d), 'r') as f:
                    existing_results = json.load(f)
                    exist_list = []
                    if "werm_ate" in existing_results and "werm" not in redo_list:
                        w_ate = existing_results["werm_ate"]
                        exist_list.append("werm")
                    if "naive_nn_ate" and "naive_nn_kl" in existing_results and "naive" not in redo_list:
                        naive_ate = existing_results["naive_nn_ate"]
                        naive_kl = existing_results["naive_nn_kl"]
                        exist_list.append("naive")
                    if len(exist_list) >= 2:
                        print("Results for {} already exist.".format(experiment))
                        print('[done]', d)
                        return
                    elif len(exist_list) > 0:
                        print("Results already exist for: {}".format(exist_list))

            if os.path.isfile("{}/results.json".format(d)):
                print("[running]", d)
                exp_args = dict()
                for pair in experiment.split('-'):
                    p = pair.split('=')
                    exp_args[p[0]] = p[1]
                graph = "dat/cg/{}.cg".format(exp_args["graph"])
                data = T.load("{}/dat.th".format(d))
                try:
                    if w_ate is None:
                        w_ate = float(werm_ate(data, graph))
                        print("WERM done: {}".format(w_ate))
                    if naive_ate is None:
                        if exp_args["graph"] in ["bow", "iv"]:
                            naive_ate = np.nan
                            naive_kl = np.nan
                        else:
                            graph_dir = "dat/cg/{}.cg".format(exp_args["graph"])
                            g = CausalGraph.read(graph_dir)
                            gen_model = CTM(graph_dir,
                                            v_size={v: 1 if v in ('X', 'Y') else exp_args["dim"] for v in g}).eval()
                            models = T.load("{}/best.th".format(d))
                            new_state_dict = OrderedDict()
                            for key in models:
                                if key[0:3] == "ctm":
                                    new_state_dict[key[4:]] = models[key]
                            gen_model.load_state_dict(new_state_dict)
                            naive_ate, naive_kl = naive_nn_tv(data, device=dev, return_kl=True, true_model=gen_model)
                            naive_ate = float(naive_ate)
                            naive_kl = float(naive_kl)
                        print("Naive NN done: {}".format(naive_ate))
                        print("Naive KL: {}".format(naive_kl))
                except:
                    print("Error with {}!".format(experiment))
                    raise

                ate_results = dict()
                with open("{}/results.json".format(d)) as f:
                    results = json.load(f)
                    ate_results["true_ate"] = results["true_ate"]
                    ate_results["ncm_ate"] = results["ncm_ate"]
                    ate_results["werm_ate"] = w_ate
                    ate_results["naive_nn_ate"] = naive_ate
                    ate_results["naive_nn_kl"] = naive_kl
                    ate_results["err_ncm_ate"] = results["err_ncm_ate"]
                    ate_results["err_werm_ate"] = results["true_ate"] - w_ate
                    ate_results["err_naive_nn_ate"] = results["true_ate"] - naive_ate

                with open("{}/ate_results.json".format(d), 'w') as f:
                    json.dump(ate_results, f)

                print("Finished {}.".format(experiment))
        except Exception:
            print("[failed]")
            raise


def send_email(subject, content=None, sender=None, recipient=None):
    if sender is None:
        sender = f'no-reply@{gethostname()}.cs.columbia.edu'
    if recipient is None:
        recipient = os.environ['USER'] + '@columbia.edu'

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient
    if content is not None:
        msg['Content'] = content

    s = SMTP('localhost')
    s.send_message(msg)
    s.quit()


if __name__ == '__main__':
    # define argparse types
    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

    def comma_separated_list(type):
        def _(string):
            if string in ('', 'none'):
                return
            else:
                return list(map(type, string.split(',')))
        return _

    def file_type(string):
        if not os.path.exists(string):
            parser.error("The file %s does not exist!" % string)
        else:
            return string

    def pipeline_type(string):
        cls = getattr(pipeline, string)
        assert issubclass(cls, pipeline.BasePipeline)
        return cls

    pipeline_choices = pipeline.BasePipeline.__subclasses__()

    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', '-p', type=pipeline_type, choices=pipeline_choices,
                        default='NLLNCMPipeline',
                        help=('the pipeline to be used for effect estimation.'
                              ' (default: %(default)s)'))
    parser.add_argument('--graph', '-G', type=file_type,
                        default='dat/cg/backdoor.cg',
                        help=('the path to the causal graph file specified in --graph-dir,'
                              ' excluding the .cg extension. (default: %(default)s)'))
    parser.add_argument('--n-samples', '-n', type=int,
                        default=1000000,
                        help=('number of samples to generate for training.'
                              ' (default: %(default)s)'))
    parser.add_argument('--dim', '-d', type=int,
                        default=1,
                        help=('number of (binary) dimensions to use for variables other'
                              ' than the treatment and outcome. (default: %(default)s)'))
    parser.add_argument('--trial-index', '-i', type=int,
                        default=0,
                        help=('integer used solely for choosing a different seed'
                              ' for experimental reproduceability. (default: %(default)s)'))
    parser.add_argument('--gpu', '-g', default=[0], type=comma_separated_list(int),
                        help=('a comma-separated list of GPU indices to use,'
                              ' or `none` for no gpus.'))
    args = parser.parse_args()

    run(args.pipeline, args.graph, args.n_samples, args.dim, args.trial_index, gpu=args.gpu,
        lockinfo=os.environ.get('SLURM_JOB_ID', ''))
