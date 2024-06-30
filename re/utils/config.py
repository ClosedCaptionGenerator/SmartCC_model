import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Capsule Network')

    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--train-data', type=str, required=True, default='/mnt/data_model/train/', help='path to the training data')
    parser.add_argument('--val-data', type=str, required=True, default='/mnt/data_model/val/', help='path to the validation data')

    return parser.parse_args()
