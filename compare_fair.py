import torch
from search_fair import valid, flops, model, args, device, get_loader
from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
from loguru import logger
from train import main
import os


def pop2rst(pop:List[List[int]], loader)-> Tuple[List[float]]:
    acc_list = []
    flops_list = []
    supernet = get_supernet()
    for i, choice in enumerate(pop):
        net  = supernet.sample(choice)
        acc =  main(net)["best_acc"]
        ff = flops(net)/1e6
        acc_list.append(acc)
        flops_list.append(ff)
        logger.info(
            "[{}/{}], choice {}, acc {}, flops {}",
            i+1, len(pop), choice, acc, ff)
    return acc_list, flops_list


def get_supernet():
    torch.manual_seed(123)
    np.random.seed(123)
    supernet = model.__dict__[args.model]().to(device)
    if args.use_pretrained:
        supernet.load_state_dict(
            torch.load(args.load_path) ["state_dict"])
    return supernet


if __name__ == '__main__':
    loader = get_loader(False)  # use test set as compare set here !
    supernet = get_supernet()

    # spawn
    ga_pop = torch.load(args.pop_path)
    ga_X = np.array(list(map(lambda x: x.X, ga_pop))).tolist()
    ga_rst = pop2rst(ga_X, loader)
    rand_X = [supernet.random_choice for i in range(100)]
    rand_rst = pop2rst(rand_X, loader)

    logger.success(ga_rst)
    logger.success(rand_rst)
    # draw
    names = ["Acc", "Flops(M)"]
    suffix = "-use_pretrained" if args.use_pretrained else ""
    title =  f"FairNas-{args.model}{suffix}"
    plt.scatter(np.array(ga_rst[0]), np.array(ga_rst[1]), c="red", alpha=0.6)
    plt.scatter(np.array(rand_rst[0]), np.array(rand_rst[1]), c="green", alpha=0.6)
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.title(title)
    os.makedirs("saved-figure", exist_ok=True)
    plt.savefig(f"saved-figure/{title}.png")
