import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
import wandb
from torch import nn, optim
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from loss.loss import FirstStageLoss
from cvivit import VIVIT
from cvivit3D import VIVIT3D
from vivq import VIVQ
from utils import get_dataloader
import random

class LUNA_Dataset(Dataset):
     def __init__(
         self,
         img_folder,
         length = 100

     ):
         super().__init__()
         self.len = length
         self.img_folder = img_folder

     def __len__(self):
         return self.len

     def __getitem__(self, idx):

         nodule = []
         while np.asarray(nodule).size == 0:
             rand_idx = random.randint(0,len(os.listdir(self.img_folder)))
             file = os.path.join(self.img_folder, os.listdir(self.img_folder)[rand_idx])
             img = np.load(file)["vol"]
             nodule = np.load(file)["nodules"]

         # print(img.shape)
         # print(nodule)
         nodule = nodule[0]
         image = img[nodule[0]-8:nodule[0]+8,nodule[1]-32:nodule[1]+32,nodule[2]-32:nodule[2]+32]
         image = torch.tensor(image[None,None,...], dtype=torch.float32)
         
         image2 = torch.tensor(np.zeros((1,1,16,64,64)), dtype=torch.float32)
         image2[:,:,:image.shape[2],:image.shape[3],:image.shape[4]] = image
         
         caption = 'lungs nodule'
         return image2


def train(proc_id, args):
    if os.path.exists(f"results/{args.run_name}/log.pt"):
        resume = True  # TODO: change back
    else:
        resume = False
    if not proc_id and args.node_id == 0:
        # if resume:
        #     wandb.init(project="Phenaki", name=args.run_name, entity="wand-tech", config=vars(args))
        # else:
        #     wandb.init(project="Phenaki", name=args.run_name, entity="wand-tech", config=vars(args))
        print(f"Starting run '{args.run_name}'....")
        print(f"Batch Size check: {args.n_nodes * args.batch_size * args.accum_grad * len(args.devices)}")
    parallel = len(args.devices) > 1
    device = torch.device(proc_id)

    if parallel:
        torch.cuda.set_device(proc_id)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend="nccl", init_method="file:///fsx/mas/phenaki/dist_file",
                                world_size=args.n_nodes * len(args.devices),
                                rank=proc_id + len(args.devices) * args.node_id)
        torch.set_num_threads(6)

    if args.model == "vivit":
        model = VIVIT(latent_size=16, compressed_frames=5, patch_size=(2, 8, 8), codebook_size=args.codebook_size).to(device)
    if args.model == "vivit3d":
        model = VIVIT3D(latent_size=64, patch_size=(2, 8, 8), codebook_size=args.codebook_size).to(device)

    elif args.model == "vivq":
        model = VIVQ(codebook_size=args.codebook_size).to(device)
    else:
        raise NotImplementedError()

    if not proc_id and args.node_id == 0:
        print(f"Number of Parameters: {sum([p.numel() for p in model.parameters()])}")

    criterion = FirstStageLoss(device=device)
    lr = 3e-4
    #dataset = get_dataloader(args)
    dataset = LUNA_Dataset(img_folder="/data/datasets/LUNA/npz/")
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    optimizer_discriminator = optim.AdamW(criterion.discriminator.parameters(), lr=lr*1e-2)

    if parallel:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    if not proc_id and args.node_id == 0:
        # wandb.watch(model)
        os.makedirs(f"results/{args.run_name}", exist_ok=True)
        os.makedirs(f"models/{args.run_name}", exist_ok=True)

    grad_accum_steps = args.accum_grad
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=args.total_steps, pct_start=0.1, div_factor=25, final_div_factor=1 / 25, anneal_strategy='linear')

    if resume:
        if not proc_id and args.node_id == 0:
            print("Loading last checkpoint....")
        logs = torch.load(f"results/{args.run_name}/log.pt")
        start_step = logs["step"] + 1
        model.load_state_dict(torch.load(f"models/{args.run_name}/model.pt", map_location=device))
        if not proc_id and args.node_id == 0:
            print("Loaded model....")
        opt_state = torch.load(f"models/{args.run_name}/optim.pt", map_location=device)
        last_lr = opt_state["param_groups"][0]["lr"]
        with torch.no_grad():
            while last_lr > optimizer.param_groups[0]["lr"]:
                scheduler.step()
        if not proc_id and args.node_id == 0:
            print(f"Initialized scheduler")
            print(f"Sanity check => Last-LR: {last_lr} == Current-LR: {optimizer.param_groups[0]['lr']} -> {last_lr == optimizer.param_groups[0]['lr']}")
        optimizer.load_state_dict(opt_state)
        del opt_state
    else:
        start_step = 0

    model.train()
    # images = torch.randn(1, 3, 128, 128)
    # videos = torch.randn(1, 10, 3, 128, 128)
    images = next(iter(dataset))
    # print("IMAGE shape :", images.shape)
    # while images.shape != (1,1,16,64,64):
    #     images = next(iter(dataset))
    #     print("IMAGE shape :", images.shape)

    pbar = tqdm(enumerate(dataset, start=start_step), total=args.total_steps, initial=start_step) if args.node_id == 0 and proc_id == 0 else enumerate(dataset, start=start_step)
    # pbar = tqdm(range(1000000))
    for step, images in pbar:
    # for step in pbar:
        images = images.to(device)
        recon = model(images)
        loss, d_loss = criterion(images, recon, step)
        loss_adjusted = loss / grad_accum_steps
        d_loss_adjusted = d_loss / grad_accum_steps

        loss_adjusted.backward()
        d_loss_adjusted.backward()
        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer_discriminator.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_discriminator.zero_grad()

        if not proc_id and args.node_id == 0:
            pbar.set_postfix({
                'loss': loss.item(),
                'd_loss': d_loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
            # wandb.log({
            #     "loss": loss,
            #     "d_loss": d_loss,
            #     "lr": optimizer.param_groups[0]['lr'],
            # })

        if args.node_id == 0 and proc_id == 0 and step % args.log_period == 0:

            orig = images
            # recon = recon[0]
            
            print("orig :", orig.shape)
            print("recon :", recon.shape)
            comp = vutils.make_grid(torch.cat([orig[:,:,orig.shape[2]//2,...], recon[:,:,orig.shape[2]//2,...]])).detach().cpu()
            # plt.imshow(comp.permute(1, 2, 0))
            # plt.show()
            vutils.save_image(comp, f"results/{args.run_name}/{step}.jpg")

            if step % args.extra_ckpt == 0:
                torch.save(model.state_dict(), f"models/{args.run_name}/model_{step}.pt")
                torch.save(optimizer.state_dict(), f"models/{args.run_name}/model_{step}_optim.pt")
            torch.save(model.state_dict(), f"models/{args.run_name}/model.pt")
            torch.save(optimizer.state_dict(), f"models/{args.run_name}/optim.pt")
            torch.save({'step': step}, f"results/{args.run_name}/log.pt")


def launch(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in args.devices])
    if len(args.devices) == 1:
        train(0, args)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "33751"
        p = mp.spawn(train, nprocs=len(args.devices), args=(args,))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "vivq_8192_drop_video"
    args.model = "vivit3d"
    args.dataset = "first_stage"
    # args.dataset_path = "file:./data/6.tar"
    args.dataset_path = "/fsx/mas/phenaki/data/raw_data/Moments_in_Time_Raw/tar_files/{0..363}.tar"
    args.total_steps = 5_000_000
    args.batch_size = 10
    args.num_workers = 10
    args.log_period = 100
    args.extra_ckpt = 10_000
    args.accum_grad = 1

    args.codebook_size = 8192
    args.clip_len = 10
    args.skip_frames = 5

    args.n_nodes = 1
    #args.node_id = int(os.environ["SLURM_PROCID"])
    args.node_id = 0
    args.devices = [0]
    # args.devices = [0]

    print("Launching with args: ", args)
    launch(
        args
    )
