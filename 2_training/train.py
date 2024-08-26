import argparse
import datetime
import os
import random
import warnings
warnings.filterwarnings("ignore")

try:
    import configparser as configparser
except ImportError:
    import ConfigParser as configparser

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import sys
sys.path.insert(1, '../common')
from models import *
from utils import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.gradcheck = False
torch.backends.cudnn.benchmark = True

start_time = datetime.datetime.now()
print("start:", start_time.strftime("%m/%d/%Y, %H:%M:%S"))

conf_parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=False,
)
conf_parser.add_argument("-c", "--config", help="Specify config file", metavar="FILE")

parser = argparse.ArgumentParser()

# Specify Training Data
parser.add_argument("--data", nargs="+", help="Path to training data")
parser.add_argument("--val_data", required=True, help="Path to validation data")
parser.add_argument("--use_s3", action="store_true", help="Use s3 buckets for training data")
parser.add_argument("--train_buckets", nargs="+", default=[], help="s3 buckets containing training data.")
parser.add_argument("--endpoint", "--endpoint_url", type=str, default=None)

# Specify Training Object
parser.add_argument("--object", nargs="+", required=True, default=[], help='Object to train network for.')
parser.add_argument("--workers", type=int, help="number of data loading workers", default=8)
parser.add_argument("--batchsize", "--batch_size", type=int, default=32, help="input batch size")
parser.add_argument("--imagesize", type=int, default=512, help="the height / width of the input image to network")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate, default=0.0001")
parser.add_argument("--net_path", default=None, help="path to net (to continue training)")
parser.add_argument("--namefile", default="epoch", help="name to put on the file of the save weights")
parser.add_argument("--manualseed", type=int, help="manual seed")
parser.add_argument("--epochs", "--epoch", "-e", type=int, default=60, help="Number of epochs to train for")
parser.add_argument("--loginterval", type=int, default=100)
parser.add_argument("--gpuids", nargs="+", type=int, default=[0], help="GPUs to use")
parser.add_argument("--exts", nargs="+", type=str, default=["png"], help="Extensions for images to use.")
parser.add_argument("--outf", default="output_real/", help="folder to output images and model checkpoints")
parser.add_argument("--sigma", default=4, help="keypoint creation sigma")
parser.add_argument("--local-rank", type=int, default=0)
parser.add_argument("--save", action="store_true", help="save a batch and quit")
parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights. Must also specify --net_path.")
parser.add_argument("--nbupdates", default=None, help="nb max update to network")

# Read the config but do not overwrite the args written
args, remaining_argv = conf_parser.parse_known_args()
defaults = {"option": "default"}

if args.config:
    config = configparser.SafeConfigParser()
    config.read([args.config])
    defaults.update(dict(config.items("defaults")))

parser.set_defaults(**defaults)
parser.add_argument("--option")
opt = parser.parse_args(remaining_argv)

local_rank = opt.local_rank

# Output to the object folder:
if opt.object:
    object_name = opt.object[0]
    opt.outf = f"{opt.outf}/{object_name}"
else:
    opt.outf = f"{opt.outf}"

os.makedirs(opt.outf, exist_ok=True)
os.makedirs(os.path.join(opt.outf, "weights"), exist_ok=True)
os.makedirs(os.path.join(opt.outf, "graphs"), exist_ok=True)

# Validate Arguments
if opt.use_s3 and (opt.train_buckets is None or opt.endpoint is None):
    raise ValueError("--train_buckets and --endpoint must be specified if training with data from s3 bucket.")

if not opt.use_s3 and opt.data is None:
    raise ValueError("--data field must be specified.")

with open(opt.outf + "/header.txt", "w") as file:
    file.write(str(opt) + "\n")

if opt.manualseed is None:
    opt.manualseed = random.randint(1, 10000)

with open(opt.outf + "/header.txt", "w") as file:
    file.write(str(opt))
    file.write("seed: " + str(opt.manualseed) + "\n")

if local_rank == 0:
    writer = SummaryWriter(opt.outf + "/runs/")

random.seed(opt.manualseed)
torch.manual_seed(opt.manualseed)
torch.cuda.manual_seed_all(opt.manualseed)

# Data Augmentation
if not opt.save:
    contrast = 0.2
    brightness = 0.2
    noise = 0.1
    normal_imgs = [0.59, 0.25]
    transform = transforms.Compose([
        AddRandomContrast(contrast),
        AddRandomBrightness(brightness),
        transforms.Resize(opt.imagesize),
    ])
else:
    contrast = 0.00001
    brightness = 0.00001
    noise = 0.00001
    normal_imgs = None
    transform = transforms.Compose([
        transforms.Resize(opt.imagesize),
        transforms.ToTensor()
    ])

# Load Model
net = DopeNetwork()
output_size = 50
opt.sigma = 0.5

train_dataset = CleanVisiiDopeLoader(
    opt.data, sigma=opt.sigma, output_size=output_size, objects=opt.object,
    use_s3=opt.use_s3, buckets=opt.train_buckets, endpoint_url=opt.endpoint
)
trainingdata = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=opt.workers, pin_memory=True
)

val_dataset = CleanVisiiDopeLoader(
    opt.val_data, sigma=opt.sigma, output_size=output_size, objects=opt.object
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=opt.batchsize, shuffle=False, num_workers=opt.workers, pin_memory=True
)

print("training data: {} batches".format(len(trainingdata)))
print("validation data: {} batches".format(len(val_loader)))

print("Loading Model...")
net = net.cuda()

if opt.pretrained:
    if opt.net_path is not None:
        net.load_state_dict(torch.load(opt.net_path))
    else:
        print("Error: Did not specify path to pretrained weights.")
        quit()

parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.Adam(parameters, lr=opt.lr)

print("ready to train!")

nb_update_network = 0
best_results = {"epoch": None, "passed": None, "add_mean": None, "add_std": None}

def has_nan(tensor):
    return torch.isnan(tensor).any().item()

def validate(model, val_loader):
    model.eval()
    val_loss = 0
    val_loss_log = []
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validating", leave=False):
            inputs = data['img'].cuda()
            target_belief = data['beliefs'].cuda()
            target_affinities = data['affinities'].cuda()

            output_belief, output_aff = model(inputs)

            loss_belief = torch.tensor(0).float().cuda()
            loss_affinities = torch.tensor(0).float().cuda()

            for stage in range(len(output_aff)):
                loss_affinities += ((output_aff[stage] - target_affinities) * (output_aff[stage] - target_affinities)).mean()
                loss_belief += ((output_belief[stage] - target_belief) * (output_belief[stage] - target_belief)).mean()

            loss = loss_affinities + loss_belief

            if has_nan(loss):
                print(f"NaN detected in validation loss")
                continue

            val_loss += loss.item()
            val_loss_log.append(loss.item())

    return val_loss / len(val_loader), val_loss_log

def _runnetwork(epoch, train_loader):
    global nb_update_network
    net.train()
    loss_avg_to_log = {
        "loss": [],
        "loss_affinities": [],
        "loss_belief": []
    }

    pbar = tqdm(train_loader, desc=f"Train Epoch: {epoch}", unit="batch")
    for batch_idx, targets in enumerate(pbar):
        optimizer.zero_grad()
        data = Variable(targets["img"].cuda())
        target_belief = Variable(targets["beliefs"].cuda())
        target_affinities = Variable(targets["affinities"].cuda())
        output_belief, output_aff = net(data)

        loss_belief = torch.tensor(0).float().cuda()
        loss_affinities = torch.tensor(0).float().cuda()

        for stage in range(len(output_aff)):
            loss_affinities += ((output_aff[stage] - target_affinities) * (output_aff[stage] - target_affinities)).mean()
            loss_belief += ((output_belief[stage] - target_belief) * (output_belief[stage] - target_belief)).mean()

        loss = loss_affinities + loss_belief

        if has_nan(loss):
            print(f"NaN detected in loss at epoch {epoch}, batch {batch_idx}")
            continue

        loss.backward()
        optimizer.step()
        nb_update_network += 1

        # Log the loss values
        loss_avg_to_log["loss"].append(loss.item())
        loss_avg_to_log["loss_affinities"].append(loss_affinities.item())
        loss_avg_to_log["loss_belief"].append(loss_belief.item())
        
        # Update the progress bar with the current loss
        pbar.set_postfix(loss=loss.item())

        if batch_idx % opt.loginterval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.15f} Local Rank: {local_rank}")

    return loss_avg_to_log

def plot_epoch_losses(epoch, train_loss_log, val_loss_log, opt):
    # Save loss graphs for the entire epoch
    graphs_dir = os.path.join(opt.outf,"graphs", f"epoch_{epoch}")
    os.makedirs(graphs_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_loss_log["loss"], label="Train Loss")
    plt.plot(val_loss_log, label="Val Loss")
    plt.title("Losses for Epoch {}".format(epoch))
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(graphs_dir, "epoch_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(train_loss_log["loss_affinities"], label="Train Affinities Loss")
    plt.title("Train Affinities Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(graphs_dir, "train_aff_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(train_loss_log["loss_belief"], label="Train Belief Loss")
    plt.title("Train Belief Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(graphs_dir, "train_bel_loss.png"))
    plt.close()

    # Save loss across all epochs
    all_epochs_dir = f"{opt.outf}/graphs/all_epochs"
    os.makedirs(all_epochs_dir, exist_ok=True)

    # Create a list to store the average loss for each epoch
    epoch_avg_losses = {
        "train": [],
        "val": []
    }

    # Calculate the average loss for the current epoch
    epoch_avg_loss_train = sum(train_loss_log["loss"]) / len(train_loss_log["loss"])
    epoch_avg_loss_val = sum(val_loss_log) / len(val_loss_log)
    epoch_avg_losses["train"].append(epoch_avg_loss_train)
    epoch_avg_losses["val"].append(epoch_avg_loss_val)

    # If this is not the first epoch, load the previous epoch's losses
    if start_epoch > 1:
        previous_train_loss_path = os.path.join(all_epochs_dir, "epoch_avg_losses_train.npy")
        previous_val_loss_path = os.path.join(all_epochs_dir, "epoch_avg_losses_val.npy")

        # Check if the files exist before loading
        if os.path.exists(previous_train_loss_path) and os.path.exists(previous_val_loss_path):
            previous_losses_train = np.load(previous_train_loss_path)
            previous_losses_val = np.load(previous_val_loss_path)

            # Append previous losses to the current epoch losses
            epoch_avg_losses["train"] = list(previous_losses_train[start_epoch-2:]) + epoch_avg_losses["train"]
            epoch_avg_losses["val"] = list(previous_losses_val[start_epoch-2:]) + epoch_avg_losses["val"]
        else:
            print(f"Previous losses not found. Starting from epoch {start_epoch} without loading previous data.")

    # if epoch > 1:
    #     previous_losses_train = np.load(os.path.join(all_epochs_dir, "epoch_avg_losses_train.npy"))
    #     previous_losses_val = np.load(os.path.join(all_epochs_dir, "epoch_avg_losses_val.npy"))
    #     epoch_avg_losses["train"] = list(previous_losses_train) + epoch_avg_losses["train"]
    #     epoch_avg_losses["val"] = list(previous_losses_val) + epoch_avg_losses["val"]

    # Save the list of average losses for all epochs
    np.save(os.path.join(all_epochs_dir, "epoch_avg_losses_train.npy"), np.array(epoch_avg_losses["train"]))
    np.save(os.path.join(all_epochs_dir, "epoch_avg_losses_val.npy"), np.array(epoch_avg_losses["val"]))

    # Plot the loss across all epochs
    plt.figure()
    plt.plot(range(1, epoch + 1), epoch_avg_losses["train"], label="Train Loss")
    plt.plot(range(1, epoch + 1), epoch_avg_losses["val"], label="Val Loss")
    plt.title("Losses across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(all_epochs_dir, "loss_across_epochs.png"))
    plt.close()

start_epoch = 1
if opt.pretrained and opt.net_path is not None:
    # we started with a saved checkpoint, we start numbering checkpoints after the loaded one
    start_epoch = int(os.path.splitext(os.path.basename(opt.net_path).split('_')[2])[0]) + 1
    print(f"Starting at epoch {start_epoch}")

for epoch in range(start_epoch, opt.epochs + 1):
    train_loss_log = _runnetwork(epoch, trainingdata)
    val_loss, batch_val_loss_log = validate(net, val_loader)
    print(f"====> Epoch: {epoch} Validation loss: {val_loss:.6f}")

    try:
        if local_rank == 0:
            torch.save(net.state_dict(), f"{opt.outf}/weights/net_{opt.namefile}_{str(epoch).zfill(2)}.pth")
    except Exception as e:
        print(f"Encountered Exception: {e}")

    if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
        break
        
    plot_epoch_losses(epoch, train_loss_log, batch_val_loss_log, opt)


print("end:", datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
print("Total time taken: ", str(datetime.datetime.now() - start_time).split(".")[0])
