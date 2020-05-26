import torch
from search_fair import valid, flops, model, args, device, get_loader
from matplotlib import pyplot as plt
import numpy as np
from loguru import logger
import os


if __name__ == '__main__':
    ga_pop = torch.load(args.pop_path)
    ga_F_acc = np.array(list(map(lambda x: -x.F[0], ga_pop)))
    ga_F_flops = np.array(list(map(lambda x: x.F[1]/1e6, ga_pop)))
    ga_rst = [ga_F_acc, ga_F_flops]

    names = ["Acc", "Flops(M)"]
    title =  f"FairNas-analyze"
    plt.scatter(np.array(ga_rst[0]), np.array(ga_rst[1]), c="red", alpha=0.6)
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.title(title)
    os.makedirs("saved-figure", exist_ok=True)
    plt.savefig(f"saved-figure/{title}.png")
