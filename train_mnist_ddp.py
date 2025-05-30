import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import wandb
import torch.multiprocessing as mp

from tqdm import tqdm

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    # Set CUDA device first
    torch.cuda.set_device(rank)
    
    # DDP setup
    if rank == 0:
        print(f"Setting up DDP with world size {world_size}")
    setup_ddp(rank, world_size)
    
    # Initialize wandb only on main process
    if rank == 0:
        wandb.login()
        wandb.init(
            project="mnist-ddp",
            config={
                "epochs": 10,
                "batch_size": 64,
                "learning_rate": 0.01,
                "architecture": "CNN",
            }
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load dataset
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=64, sampler=sampler, drop_last=True)
    
    # Create model and wrap with DDP
    model = ConvNet().to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Synchronize all processes before starting training
    dist.barrier()
    if rank == 0:
        print("Starting training")

    # Training loop
    model.train()
    for epoch in range(10):
        sampler.set_epoch(epoch)
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Only show progress bar on rank 0
        if rank == 0:
            iterator = tqdm(enumerate(train_loader), total=len(train_loader))
        else:
            iterator = enumerate(train_loader)

        for batch_idx, (data, target) in iterator:
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            running_loss += loss.item()

            if batch_idx % 100 == 0 and rank == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Log metrics on main process
        if rank == 0:
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            wandb.log({
                "epoch": epoch,
                "loss": epoch_loss,
                "accuracy": epoch_acc
            })
            print(f'Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%')

        # Synchronize at the end of each epoch
        dist.barrier()

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"
    
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    ) 