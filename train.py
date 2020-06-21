import sys
import os
import warnings
from models.VGGnet import vggnet
from utils.utils import save_checkpoint
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import argparse
import json
import cv2
from data.data_loader import ImageDataset
import time

parser = argparse.ArgumentParser(description='PyTorch Multi-granularity-integration3 using VGG16')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str, help='path to the pretrained model')
args = parser.parse_args()

args.original_lr = 1e-5
args.lr = 1e-5
args.batch_size = 32
args.momentum = 0.95
args.weight_decay = 5 * 1e-4
args.start_epoch = 0
args.epochs = 1000
args.workers = 4
args.print_freq = 400
args.gpu_id = "cuda:1"

args.dataset_path = '/home/njuciairs/rainkeeper/Projects/Datasets/crop_image'
args.seed = time.time()

def main():
    cut_list = [1, 2, 3]
    for i in range(len(cut_list)):
        if i < 2:
            continue
        for j in range(cut_list[i] * cut_list[i]):
            if j < 7:
                continue
            args.best_test_precision = 0
            args.best_epoch = 0
            checkpoint_save_dir = '/home/njuciairs/rainkeeper/Projects/PycharmProjects/Multi-granularity-integrations3/checkpoint0/' + str(cut_list[i] * cut_list[i]) + '_' + str(j)
            if not os.path.exists(checkpoint_save_dir):
                os.makedirs(checkpoint_save_dir)
            terminal_log_file = os.path.join(checkpoint_save_dir, 'terminal_log.txt')
            terminal_file = open(terminal_log_file, 'a')

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            trainset = ImageDataset(image_dir=os.path.join(args.dataset_path, 'train_image', str(cut_list[i] * cut_list[i]) + '_' + str(j)),
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize,
                                    ]),
                                    train=True)
            train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                                                       num_workers=args.workers)

            valset = ImageDataset(image_dir=os.path.join(args.dataset_path, 'test_image', str(cut_list[i] * cut_list[i]) + '_' + str(j)),
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      normalize,
                                  ]),
                                  train=False)
            val_loader = torch.utils.data.DataLoader(valset, shuffle=False, batch_size=args.batch_size,
                                                     num_workers=args.workers)

            args.device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")
            torch.cuda.manual_seed(args.seed)

            model = vggnet(load_weights=True)
            model.to(args.device)

            criterion = nn.CrossEntropyLoss().to(args.device)
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

            if args.pre:
                if os.path.isfile(args.pre):
                    print("=> loading checkpoint '{}'".format(args.pre))
                    checkpoint = torch.load(args.pre)
                    args.start_epoch = checkpoint['epoch']
                    args.best_test_precision = checkpoint['best_test_precision']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                          .format(args.pre, checkpoint['epoch']))
                else:
                    print("=> no checkpoint found at '{}'".format(args.pre))

            for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
                train_precision = train(train_loader, model, criterion, optimizer, epoch, terminal_file)
                test_correct, test_total = validate(val_loader, model, criterion, terminal_file)
                test_precision = float(test_correct) / test_total

                is_best = test_precision > args.best_test_precision
                if is_best:
                    args.best_test_precision = test_precision
                    args.best_epoch = epoch

                print('Epoch:%d, test correct:%d, test total:%d, test precsion:%.6f, current_best_test_acc:%.6f, current_best_acc_epoch:%d' % (epoch, test_correct, test_total, test_precision, args.best_test_precision, args.best_epoch))
                print('Epoch:%d, test correct:%d, test total:%d, test precsion:%.6f, current_best_test_acc:%.6f, current_best_acc_epoch:%d' % (epoch, test_correct, test_total, test_precision, args.best_test_precision, args.best_epoch), file=terminal_file)

                if is_best:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.pre,
                        'state_dict': model.state_dict(),
                        'best_test_precision': args.best_test_precision,
                        'optimizer': optimizer.state_dict()
                    }, checkpoint_save_dir, epoch)

                if train_precision > 0.98:
                    print('best_test_precision:%.6f' % args.best_test_precision)
                    print('best_test_precision:%.6f' % args.best_test_precision, file=terminal_file)
                    break


def train(train_loader, model, criterion, optimizer, epoch, terminal_file):
    correct = 0.0
    total = 0.0

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    end = time.time()

    for i, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(args.device)
        outputs = model(images)
        labels = labels.to(args.device)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_correct, batch_total = accuracy(outputs, labels)
        correct += batch_correct
        total += batch_total

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses), file=terminal_file)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
    print('train correct:%d, train total:%d, train_precision:%.6f' % (int(correct), int(total), float(correct) / total))
    print('train correct:%d, train total:%d, train_precision:%.6f' % (int(correct), int(total), float(correct) / total), file=terminal_file)
    train_precision = float(correct) / total
    return train_precision


def validate(val_loader, model, criterion, terminal_file):
    print('begin test')
    print('begin test', file=terminal_file)

    model.eval()
    correct = 0.0
    total = 0.0

    for i, (images, labels) in enumerate(val_loader):
        images = images.to(args.device)
        outputs = model(images)
        labels = labels.to(args.device)

        batch_correct, batch_total = accuracy(outputs, labels)
        correct += batch_correct
        total += batch_total

    # print('test correct:%d, test total:%d, test_precsion:%.6f' % (correct, total, correct/total))
    return correct, total


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs, labels):
    batch_total = labels.size(0)
    _, predicted = torch.max(outputs.data, 1)
    batch_correct = (predicted == labels).sum()
    return batch_correct, batch_total


if __name__ == '__main__':
    main()
