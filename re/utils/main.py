import argparse
from trainers.trainer import Trainer
from config import get_args

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(config, device)
    trainer.train()

if __name__ == "__main__":
    config = vars(get_args())
    main(config)
