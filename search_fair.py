import os
import numpy as np
import argparse
from utils import *
import autograd.numpy as anp
import pymoo
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
#from wnsga2 import WNSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.factory import get_termination
from pymoo.visualization.scatter import Scatter
import torch
import time
import math
from itertools import combinations
import collections
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import utils
from functools import partial
from fair_nas.search import search
from torchvision import datasets, transforms
from train import run_model, args, model, device
from loguru import logger


def valid(net, loader):
    criterion = torch.nn.CrossEntropyLoss()
    _, acc = run_model(net, loader, criterion, None, train=False)
    return -acc


def flops(net):
    flops, _ = utils.get_model_complexity_info(net, (1,28,28), False, False)
    return flops


def get_loader(train_set=True):
    """
    Attention!!!!! use train_set as search set here
    """
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if args.data == 'FashionMNIST':
        trainset = datasets.FashionMNIST('./data', train=train_set, download=True, transform=val_transforms)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.nworkers)
        print('Training on FashionMNIST')
    else:
        trainset = datasets.MNIST('./data-mnist', train=train_set, download=True, transform=val_transforms)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.nworkers)
        print('Training on MNIST')
    return train_loader


class MyProblem(Problem):
    def __init__(self, supernet):
        self.supernet = supernet
        self.obj_functions = [
            partial(valid, loader=get_loader()),
            flops
        ]
        self.obj_names = ['valid', 'flops']
        super().__init__(n_var=supernet.nl,
                         n_obj=2,
                         n_constr=0,
                         xl=anp.array([0]*supernet.nl),
                         xu=anp.array([supernet.nb-1]*supernet.nl))

    def _evaluate(self, x, out, *args, **kwargs):
        """
        max acc
        min flops
        """
        num_pop = x.shape[0]
        f_res = [np.zeros(num_pop) for _ in self.obj_functions]

        for i in range(num_pop):
            choice = x[i].tolist()
            logger.debug(choice)
            net = self.supernet.sample(choice)
            for j in range(len(self.obj_functions)):
                f_res[j][i] = self.obj_functions[j](net)
            show = ""
            for name, res in zip(self.obj_names, [res[i] for res in f_res]):
                show += "%s:%.6f, " % (name, res)
            logger.info("PopID: {} | Choice {} | Result {}", i, choice, show)
        out["F"] = anp.column_stack(f_res)


if __name__ == "__main__":
    supernet = model.__dict__[args.model]().to(device)
    supernet.load_state_dict(
        torch.load(args.load_path) ["state_dict"])
    problem = MyProblem(supernet)

    res = search(problem,
            pop_size=20,
            n_offsprings=10,
            n_generations=10,
            seed=123)
    X = np.array(list(map(lambda x: x.X, res.pop)))
    F = np.array(list(map(lambda x: x.F, res.pop)))
    logger.info("get popuation")
    logger.success(X)
    logger.info("get result")
    logger.success(F)
    os.makedirs("saved-pops", exist_ok=True)
    torch.save(res.pop, "saved-pops/fair_pop.pkl")
