import model.locator.clip as clip
import torch
import argparse

from dataset.RefcocogDataset import RefcocogDataset
from torch.utils.data import DataLoader

from FocalDiceLoss import FocalDiceLoss
import wandb
from datetime import datetime
from tqdm import tqdm
import os

############################################
# DEFINE ARGUMENTS
############################################

arg = argparse.ArgumentParser()
arg.add_argument("--name", type=str, default='run_{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), help="Name of the run")
arg.add_argument("--batch_size", type=int, default=16, help="Batch size")
arg.add_argument("--num_epochs", type=int, default=60, help="Number of epochs")
arg.add_argument("--dataset", type=str, default="./refcocog", help="Dataset to use")
arg.add_argument("-l", "--logwandb", help="Log training on wandb", action="store_true")
arg.add_argument("-r", "--resume", help="Resume Training", action="store_true")
arg.add_argument("--model", type=str, help="Model parameters to use")
arg.add_argument("--optimizer", type=str, help="Optimizer paramteres to use")
arg.add_argument("--scheduler", type=str, help="Scheduler paramteres to use")
arg.add_argument("--epoch", type=int, help="Resuming starting epoch")

args = vars(arg.parse_args())

logwandb = args["logwandb"]
resume = args["resume"]

############################################
# DEFINE TRAINING FUNCTIONS
############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

def load_scheduler(scheduler, path):
    scheduler.load_state_dict(torch.load(path))
    return scheduler

def load_optimizer(optimizer, path):
    optimizer.load_state_dict(torch.load(path))
    return optimizer


def train_one_epoch(epoch_index, train_loader, model, criterion, optimizer, loop):
    epoch_losses = [] # list of losses containing the losses for each batch
    for i, (samples, bbox) in enumerate(train_loader):
        loop.set_postfix_str(f'Batch {i+1}/{len(train_loader)}')


        images = samples['image'].to(device)
        sentences = clip.tokenize(samples['sentences']).to(device)
        target = bbox['gt'].to(device, dtype=torch.float32) # ground truth segmentation map in 16x16 resolution

        optimizer.zero_grad()

        maps, fv = model.encode(images, sentences)

        batch_loss = criterion(maps, target) # returns total loss for current batch

        batch_loss.backward()
        optimizer.step()

        epoch_losses.append(batch_loss.item())

        if logwandb:
            wandb.log({"batch_loss": batch_loss.item()})

    return torch.mean(torch.tensor(epoch_losses)).item()

def train_loop(num_epochs, train_loader, model, criterion, optimizer, scheduler, eval_loader, num_epochs_trained=0):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # used to name the model pth

    # create folder for run
    run_path = 'runs/{}'.format(args["name"])
    if not os.path.exists(run_path):
        os.system(f"mkdir {run_path}")

    best_eval_loss = float('inf') # keep track of best eval loss to save best model

    loop = tqdm(range(num_epochs_trained, num_epochs), desc="Training locator", leave=True)
    for epoch in loop:
        model.train()

        # TRAIN ONE EPOCH
        epoch_loss = train_one_epoch(epoch, train_loader, model, criterion, optimizer, loop)

        model.eval()

        # EVALUATE MODEL AFTER EACH EPOCH
        eval_losses = []
        with torch.no_grad():
            for samples, bbox in eval_loader:
                images = samples['image'].to(device)
                sentences = clip.tokenize(samples['sentences']).to(device)
                maps, _ = model.encode(images, sentences)

                batch_loss = criterion(maps, bbox['gt'].to(device, dtype=torch.float32))

                eval_losses.append(batch_loss.item())

            eval_loss = torch.mean(torch.tensor(eval_losses)).item()

            loop.write(f'Epoch {epoch+1}/{num_epochs}\tEval loss: {eval_loss:.4f}')

            if logwandb:
                wandb.log({"train_loss": epoch_loss, "eval_loss": eval_loss})

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), run_path + "/best.pth")
        
        scheduler.step()

        torch.save(model.state_dict(), run_path + "/latest.pth")
        torch.save(optimizer.state_dict(), run_path + "/optimizer_latest.pth")
        torch.save(scheduler.state_dict(), run_path + "/scheduler_latest.pth")


if __name__ == "__main__":
    ########################################
    # INITIALIZE CLIP MODEL
    ########################################

    model, preprocess = clip.load("ViT-B/16") # only works with ViT-B/16
    model.init_adapters() # add adapters to CLIP after loading its weights from pretrained model
    if resume:
        model.load_state_dict(torch.load(args["model"])) # when needed to resume training
    model.freeze_for_training() # freeze CLIP backbone except the adapters

    model = model.to(device)
    model.to(torch.float32)


    ########################################
    # INITIALIZE DATASET
    ########################################

    batch_size = args["batch_size"]
    train_dataset = RefcocogDataset(args["dataset"], split="train", transform=preprocess)
    val_dataset = RefcocogDataset(args["dataset"], split="val", transform=preprocess)
    test_dataset = RefcocogDataset(args["dataset"], split="test", transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ########################################
    # INITIALIZE TRAINING PARAMETERS
    ########################################

    learning_rate = 5e-5/ (32/args["batch_size"]) # 5e-5/2 for 16 batch size
    weight_decay = 5e-3 # 5e-3
    num_epochs = args["num_epochs"] # to change if epochs alredy trained
    num_epochs_trained = 0 # to change if epochs alredy trained
    
    if resume:
        num_epochs_trained = args["epoch"]

    ########################################
    # INITIALIZE LOSS FUNCTION, OPTIMIZER AND SCHEDULER
    ########################################

    apply_sigmoid = True
    criterion = FocalDiceLoss(apply_sigmoid=apply_sigmoid)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay, eps=1e-08)
    if resume:
        optimizer = load_optimizer(optimizer, path=args["optimizer"]) # when needed to resume training
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs)

    if resume:
        scheduler = load_scheduler(scheduler, path=args["scheduler"]) # when needed to resume training

    # specify parameters to save for logging on wandb
    if logwandb:
        wandb.init(project="projectdl", 
                name=args["name"], 
                config={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "num_epochs_trained": num_epochs_trained,
                    "focal_alpha": 0.65,
                    "focal_gamma": 2.0,
                    "lambda_focal": 1.75,
                    "lambda_dice": 1.0,
                    "sigmoid": apply_sigmoid
                    }
        )

    ########################################
    # TRAINING LOOP
    ########################################

    train_loop(num_epochs, train_loader, model, criterion, optimizer, scheduler, val_loader, num_epochs_trained)

    wandb.finish()