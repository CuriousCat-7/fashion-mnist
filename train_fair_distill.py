import torch
from torch import nn
from torchvision import datasets, models, transforms
import torch.optim as optim
import model
import utils
import time
import argparse
import os
import csv
from datetime import datetime
from loguru import logger
# from tensorboardX import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--model", type=str, default='FashionComplexNet', help="model")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--teacher-model", type=str, default='FashionComplexNet', help="model")
    parser.add_argument("--teacher-path", type=str, required=True)
    # train params
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--nepochs", type=int, default=200, help="max epochs")
    parser.add_argument("--nworkers", type=int, default=4, help="number of workers")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--data", type=str, default='FashionMNIST', help="MNIST, or FashionMNIST")
    parser.add_argument("--ngpus", type=int, default=1, help="number of gpus")
    # Search
    parser.add_argument("--n_gen", type=int, default=10)
    args = parser.parse_args()
    return args


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device== 'cuda:0':
        torch.cuda.manual_seed(args.seed)
    print('Training on {}'.format(device))
    return device


def setup(args):
    torch.manual_seed(args.seed)
    # Setup folders for saved models and logs
    if not os.path.exists('saved-models/'):
        os.mkdir('saved-models/')
    if not os.path.exists('logs/'):
        os.mkdir('logs/')

    # Setup folders. Each run must have it's own folder. Creates
    # a logs folder for each model and each run.
    out_dir = 'logs/{}'.format(args.model)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    run = 0
    current_dir = '{}/run-{}'.format(out_dir, run)
    while os.path.exists(current_dir):
        run += 1
        current_dir = '{}/run-{}'.format(out_dir, run)
    os.mkdir(current_dir)
    logfile = open('{}/log.txt'.format(current_dir), 'w')
    logger.add(logfile)
    logger.info(args)
    return current_dir, run


def get_loader(args, train_shuffle=True):
    # Define transforms.
    trival_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Create dataloaders. Use pin memory if cuda.

    if args.data == 'FashionMNIST':
        trainset = datasets.FashionMNIST('./data', train=True, download=True, transform=trival_transforms)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                  shuffle=train_shuffle, num_workers=args.nworkers)
        valset = datasets.FashionMNIST('./data', train=False, transform=trival_transforms)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.nworkers)
        print('Training on FashionMNIST')
    else:
        trainset = datasets.MNIST('./data-mnist', train=True, download=True, transform=trival_transforms)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                  shuffle=train_shuffle, num_workers=args.nworkers)
        valset = datasets.MNIST('./data-mnist', train=False, transform=trival_transforms)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.nworkers)
        print('Training on MNIST')
    return train_loader, val_loader


def get_net(args, device):
    net = model.__dict__[args.model]().to(device)
    return net


def get_teacher_net(args, device):
    teacher_net = model.__dict__[args.teacher_model]().to(device)
    teacher_net.load_state_dict(torch.load(args.teacher_path)["state_dict"])
    return teacher_net


def get_optim_fair(args, net):
    # Define optimizer
    #optimizer = optim.Adam(net.parameters())
    comm_params = list(net.stem.parameters()) +\
        list(net.tail.parameters()) +\
        list(net.classifier.parameters())
    nas_params = list(net.mid.parameters())
    params = [
        {"params": nas_params},
        {"params": comm_params, "lr":1e-3/3},
    ]
    optimizer = optim.Adam(params)
    return optimizer


def get_optim(args, net):
    optimizer = optim.Adam( net.parameters() )
    return optimizer


class RunModel():
    def __init__(self,args, net, train_loader, val_loader, optimizer, device ):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def train_epoch(self):
        return self._run_model(self.train_loader, True)

    def valid_epoch(self):
        return self._run_model(self.val_loader, False)

    def _run_model(self, loader, train = True):
        running_loss = 0.0
        running_accuracy = 0.0

        # Set mode
        if train:
            self.net.train()
        else:
            self.net.eval()

        for i, (X, y) in enumerate(loader):
            # Pass to gpu or cpu
            X, y = X.to(self.device), y.to(self.device)
            loss, pred = self._step(X, y, train)
            # Calculate stats
            running_loss += loss.item()
            running_accuracy += torch.sum(pred == y.detach()) if pred is not None else 0
        return running_loss / len(loader), running_accuracy / len(loader.dataset)

    def _step(self, X, y, train):
        with torch.set_grad_enabled(train):
            if train:
                # Zero the gradient
                self.optimizer.zero_grad()
                for choice in self.net.random_shuffle:
                    self.net.set_choice(choice)
                    output = self.net(X)
                    loss = self.criterion(output, y)
                    loss.backward()
                self.optimizer.step()
            else:
                self.net.set_choice(self.net.random_choice)
                output = self.net(X)
                loss = self.criterion(output, y)

        _, pred = torch.max(output, 1)
        return loss, pred


class RunModelDistill(RunModel):
    def __init__(self,args, teacher_net, net, train_loader, val_loader, optimizer, device ):
        self.net = net
        self.teacher_net = teacher_net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device

    def criterion(self, student_oups, teacher_oups, student_Ks):
        loss = 0
        for s_oup, t_oup, K in zip(student_oups, teacher_oups, student_Ks):
            loss += (s_oup - t_oup).pow(2).sum().div(K)
        return loss

    def _step(self, X, y, train):
        self.teacher_net.eval()
        with torch.no_grad():
            t_inps, t_oups = self.teacher_net.forward_teach(X)
        with torch.set_grad_enabled(train):
            if train:
                # Zero the gradient
                self.optimizer.zero_grad()
                for choice in self.net.random_shuffle:
                    self.net.set_choice(choice)
                    s_oups, s_Ks = self.net.forward_distill(t_inps)
                    loss = self.criterion(s_oups, t_oups, s_Ks)
                    loss.backward()
                self.optimizer.step()
            else:
                self.net.set_choice(self.net.random_choice)
                s_oups, s_Ks = self.net.forward_distill(t_inps)
                loss = self.criterion(s_oups, t_oups, s_Ks)

        pred = None
        return loss, pred


class RunModelFairDistill(RunModelDistill):

    def ce_criterion(self, output, y):
        return nn.CrossEntropyLoss()(output, y)

    def _step(self, X, y, train):
        self.teacher_net.eval()
        with torch.no_grad():
            t_inps, t_oups = self.teacher_net.forward_teach(X)
        with torch.set_grad_enabled(train):
            if train:
                # Zero the gradient
                self.optimizer.zero_grad()
                for choice in self.net.random_shuffle:
                    self.net.set_choice(choice)
                    s_oups, s_Ks = self.net.forward_distill(t_inps)
                    output = self.net(X)
                    loss = self.ce_criterion(output, y) + self.criterion(s_oups, t_oups, s_Ks)
                    loss.backward()
                self.optimizer.step()
            else:
                self.net.set_choice(self.net.random_choice)
                s_oups, s_Ks = self.net.forward_distill(t_inps)
                output = self.net(X)
                loss = self.ce_criterion(output, y) + self.criterion(s_oups, t_oups, s_Ks)

        _, pred = torch.max(output, 1)
        return loss, pred



def train_until_patience(args, run_model, run):
    flops_count , params_count = None, None
    patience = args.patience
    best_loss = 1e4
    best_acc = 0
    begin = datetime.now()
    for e in range(args.nepochs):
        start = time.time()
        train_loss, train_acc = run_model.train_epoch()
        val_loss, val_acc = run_model.valid_epoch()
        end = time.time()

        # print stats
        stats = """\nEpoch: {}\t train loss: {:.3f}, train acc: {:.3f}\t
                val loss: {:.3f}, val acc: {:.3f}\t
                time: {:.1f}s""".format(e+1, train_loss, train_acc, val_loss,
                                        val_acc, end - start)
        logger.info(stats)


        # early stopping and save best model

        if val_acc > best_acc:
            best_acc = val_acc.cpu().item()

        model_path = 'saved-models/{}-train-{}.pth.tar'.format(args.model, run)
        if val_loss < best_loss:
            best_loss = val_loss
            patience = args.patience
            utils.save_model({
                'arch': args.model,
                'state_dict': utils.get_state_dict(run_model.net)
            }, model_path)
        else:
            patience -= 1
            if patience == 0:
                print('Run out of patience!')
                break

    rst = dict(
        best_loss=best_loss,
        best_acc=best_acc,
        flops_count=flops_count,
        params_count=params_count,
        used_time = datetime.now() - begin,
        model_path = model_path,
    )
    logger.info(rst)
    return rst


def main(args):
    device = get_device()
    current_dir, run = setup(args)
    train_loader, val_loader = get_loader(args)
    net = get_net(args, device)

    ## train distill
    #teacher_net = get_teacher_net(args, device)
    #optimizer = get_optim(args, net)
    #run_distill = RunModelDistill(args, teacher_net, net, train_loader, val_loader, optimizer, device)
    #rst = train_until_patience(args, run_distill, run)
    #logger.success("distill finish, rst is {}", rst)

    # train fair distill
    teacher_net = get_teacher_net(args, device)
    optimizer = get_optim(args, net)
    run_model = RunModelFairDistill(args, teacher_net, net, train_loader, val_loader, optimizer, device)
    rst = train_until_patience(args, run_model, run)
    logger.success("distill finish, rst is {}", rst)

    # train fair
    optimizer = get_optim_fair(args, net)
    run_fair = RunModel(args, net, train_loader, val_loader, optimizer, device)
    rst = train_until_patience(args, run_fair, run)
    logger.success("fairnas finish, rst is {}", rst)



if __name__ == '__main__':
    main(get_args())
