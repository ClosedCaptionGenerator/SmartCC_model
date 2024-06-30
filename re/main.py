import json
import torch
from trainers.trainer import Trainer
from config import get_config

def main():
    config = get_config('config/config.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(config, device)
    trainer.train()

if __name__ == "__main__":
    main()
