import model.locator.clip as clip
# import clip
import torch
import argparse

from dataset.RefcocogDataset import RefcocogDataset
from torch.utils.data import DataLoader
from model.refiner.refiner import Refiner

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
arg.add_argument("--batch_size", type=int, default=30, help="Batch size")
arg.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
arg.add_argument("--dataset", type=str, default="./refcocog", help="Dataset to use")
arg.add_argument("-l", "--logwandb", help="Log training on wandb", action="store_true")

args = vars(arg.parse_args())

logwandb = args["logwandb"]

############################################
# DEFINE TRAINING FUNCTIONS
############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

def load_locator(path):
    locator, preprocess = clip.load("ViT-B/16")
    locator.init_adapters()
    locator = locator.to(device, dtype=torch.float32)
    locator.load_state_dict(torch.load(path))
    locator.eval()
    return locator, preprocess

def train_one_epoch(epoch_index, train_loader, locator, refiner, criterion, optimizer, loop):
    epoch_losses = []
    for i, (samples, bbox) in enumerate(train_loader):
        loop.set_postfix_str(f'Batch {i+1}/{len(train_loader)}')

        images = samples['image'].to(device)
        sentences = clip.tokenize(samples['sentences']).to(device)
        target =  bbox['gt_refiner'].to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            maps, fv = locator.encode(images, sentences)
        
        out = refiner(maps, fv) # maps is batchx14x14, fv is list (4 elem) of batchx197x768

        batch_loss = criterion(out, target) # fix to input needed, but should be 224x224 map for both target and out, so focal and dice are still good

        batch_loss.backward()
        optimizer.step()

        epoch_losses.append(batch_loss.item())

        if logwandb:
            wandb.log({"batch_loss": batch_loss.item()})

    return torch.mean(torch.tensor(epoch_losses)).item()

def train_loop(num_epochs, train_loader, eval_loader, locator, refiner, criterion, optimizer):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # create folder for run
    run_path = 'runs/{}'.format(args["name"])
    if not os.path.exists(run_path):
        os.system(f"mkdir {run_path}")

    best_eval_loss = float('inf')

    loop = tqdm(range(num_epochs), desc="Training locator", leave=True)
    for epoch in loop:
        refiner.train()

        # TRAIN ONE EPOCH
        epoch_loss = train_one_epoch(epoch, train_loader, locator, refiner, criterion, optimizer, loop)

        refiner.eval()

        # EVALUATE MODEL
        eval_losses = []
        with torch.no_grad():
            for samples, bbox in eval_loader:
                images = samples['image'].to(device)
                sentences = clip.tokenize(samples['sentences']).to(device)
                target =  bbox['gt_refiner'].to(device)
                
                maps, fv = locator.encode(images, sentences)
                out = refiner(maps, fv) # fix to correct input

                batch_loss = criterion(out, target) # fix to correct input

                eval_losses.append(batch_loss.item())

            eval_loss = torch.mean(torch.tensor(eval_losses)).item()

            loop.write(f'Epoch {epoch+1}/{num_epochs}\tEval loss: {eval_loss:.4f}')

            if logwandb:
                wandb.log({"train_loss": epoch_loss, "eval_loss": eval_loss})

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(refiner.state_dict(), run_path + "/best.pth")
        
        torch.save(refiner.state_dict(), run_path + "/epoch_" + str(epoch+1) + ".pth")



if __name__ == "__main__":
    ########################################
    # INITIALIZE MODELS
    ########################################

    locator, preprocess = load_locator(path="runs/DiceLossFix/latest.pth") # change path

    refiner = Refiner()
    refiner = refiner.to(device)
    refiner.to(torch.float32)

    ########################################
    # INITIALIZE DATASET
    ########################################

    batch_size = args["batch_size"]
    train_dataset = RefcocogDataset(args["dataset"], split="train", transform=preprocess)
    val_dataset = RefcocogDataset(args["dataset"], split="val", transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    ########################################
    # INITIALIZE TRAINING PARAMETERS
    ########################################

    learning_rate = 5e-5 # 5e-5
    weight_decay = 5e-3 # 5e-3
    num_epochs = args["num_epochs"]

    ########################################
    # INITIALIZE LOSS FUNCTION, OPTIMIZER AND SCHEDULER
    ########################################

    apply_sigmoid = True
    criterion = FocalDiceLoss(apply_sigmoid=apply_sigmoid)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay, eps=1e-08)
    # optimizer = load_optimizer(optimizer, path="") # when needed to resume training
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs)
    # scheduler = load_scheduler(scheduler, path="") # when needed to resume training

    if logwandb:
        wandb.init(project="projectdl", 
                name=args["name"], 
                config={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "loss_fn": "1.75*focal+dice loss",
                    "focal_alpha": 0.65,
                    "focal_gamma": 2.0,
                    "lambda_focal": 1.75,
                    "lambda_dice": 1.0,
                    "sigmoid": apply_sigmoid,
                    }
        )

    ########################################
    # TRAINING LOOP
    ########################################

    train_loop(num_epochs, train_loader, val_loader, locator, refiner, criterion, optimizer)

    wandb.finish()