import torch
from torchvision import datasets, models, transforms
import torch.optim as optim
import model
import utils
import time
import argparse
import os
import csv
from datetime import datetime
# from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='FashionComplexNetDistillNas', help="model")
parser.add_argument("--load-path", type=str, default="")
parser.add_argument("--teacher-model", type=str, default='FashionComplexNet', help="model")
parser.add_argument("--teacher-path", type=str, required=True)
parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--nepochs", type=int, default=200, help="max epochs")
parser.add_argument("--nworkers", type=int, default=4, help="number of workers")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--data", type=str, default='FashionMNIST', help="MNIST, or FashionMNIST")
args = parser.parse_args()


#viz
# tsboard = SummaryWriter()

# Set up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))

# Set seeds. If using numpy this must be seeded too.
torch.manual_seed(args.seed)
if device== 'cuda:0':
    torch.cuda.manual_seed(args.seed)

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
print(args, file=logfile)



# Define transforms.
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Create dataloaders. Use pin memory if cuda.

if args.data == 'FashionMNIST':
    trainset = datasets.FashionMNIST('./data', train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.nworkers)
    valset = datasets.FashionMNIST('./data', train=False, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.nworkers)
    print('Training on FashionMNIST')
else:
    trainset = datasets.MNIST('./data-mnist', train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.nworkers)
    valset = datasets.MNIST('./data-mnist', train=False, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.nworkers)
    print('Training on MNIST')


def run_model(teacher_net, net, loader, criterion, optimizer, train = True, is_search=False):
    running_loss = 0
    running_v_loss = 0
    teacher_net.eval()

    # Set mode
    if train:
        net.train()
    else:
        net.eval()


    for i, (X, y) in enumerate(loader):
        # Pass to gpu or cpu
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            teacher_inps, teacher_oups = teacher_net.forward_teach(X)


        with torch.set_grad_enabled(train):
            if train:
                # Zero the gradient
                optimizer.zero_grad()
                for choice in net.random_shuffle:
                    net.set_choice(choice)
                    student_oups = net.forward_distill(teacher_inps)
                    loss = 0
                    for s_oup, t_oup in zip(student_oups, teacher_oups):
                        loss += (s_oup - t_oup).pow(2).mean()
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
                optimizer.step()
                v_loss = torch.zeros(1)
            else:
                if not is_search:
                    net.set_choice(net.random_choice)
                student_oups = net.forward_distill(teacher_inps)
                loss = 0
                v_loss = 0
                for s_oup, t_oup in zip(student_oups, teacher_oups):
                    loss += (s_oup - t_oup).pow(2).mean()
                    v_loss += (s_oup - t_oup).abs().mean().div(t_oup.std())

        # Calculate stats
        running_loss += loss.item()
        running_v_loss += v_loss.item()
    return running_loss / len(loader), running_v_loss / len(loader)




def main(net, teacher_net):
    # Init network, criterion and early stopping
    flops_count , params_count = None, None
    criterion = torch.nn.CrossEntropyLoss()



    # Define optimizer
    optimizer = optim.Adam(net.parameters())

    # Train the network
    patience = args.patience
    best_loss = 1e4
    best_v_loss = 0
    writeFile = open('{}/stats.csv'.format(current_dir), 'a')
    writer = csv.writer(writeFile)
    writer.writerow(['Epoch', 'Train Loss', 'Train v_lossuracy', 'Validation Loss', 'Validation v_lossuracy'])
    begin = datetime.now()
    for e in range(args.nepochs):
        start = time.time()
        train_loss, train_v_loss = run_model(teacher_net, net, train_loader,
                                      criterion, optimizer)
        val_loss, val_v_loss = run_model(teacher_net, net, val_loader,
                                      criterion, optimizer, False)
        end = time.time()

        # print stats
        stats = """Epoch: {}\t train loss: {:.6f}, train v_loss: {:.6f}\t
                val loss: {:.6f}, val v_loss: {:.6f}\t
                time: {:.1f}s""".format(e+1, train_loss, train_v_loss, val_loss,
                                        val_v_loss, end - start)
        print(stats)

        # viz
        # tsboard.add_scalar('data/train-loss',train_loss,e)
        # tsboard.add_scalar('data/val-loss',val_loss,e)
        # tsboard.add_scalar('data/val-v_lossuracy',val_v_loss.item(),e)
        # tsboard.add_scalar('data/train-v_lossuracy',train_v_loss.item(),e)


        # Write to csv file
        writer.writerow([e+1, train_loss, train_v_loss, val_loss, val_v_loss])
        # early stopping and save best model

        if val_v_loss > best_v_loss:
            best_v_loss = val_v_loss

        if val_loss < best_loss:
            best_loss = val_loss
            patience = args.patience
            utils.save_model({
                'arch': args.model,
                'state_dict': net.state_dict()
            }, 'saved-models/{}-distill-{}.pth.tar'.format(args.model, run))
        else:
            patience -= 1
            if patience == 0:
                print('Run out of patience!')
                writeFile.close()
                # tsboard.close()
                break
    rst = dict(
        best_loss=best_loss,
        best_v_loss=best_v_loss,
        flops_count=flops_count,
        params_count=params_count,
        used_time = datetime.now() - begin
    )
    print(rst)
    return rst

if __name__ == '__main__':
    net = model.__dict__[args.model]().to(device)
    teacher_net = model.__dict__[args.teacher_model]().to(device)
    teacher_net.load_state_dict(torch.load(args.teacher_path)["state_dict"])
    main(net, teacher_net)
