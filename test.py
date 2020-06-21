import torch
import numpy as np
import sys
from torchvision import datasets, transforms
import os
import time
import re
import glob

from data.data_loader import ImageDataset
from models.VGGnet import vggnet

def accuracy(outputs, labels):
    batch_total = labels.size(0)
    _, predicted = torch.max(outputs.data, 1)
    batch_correct = (predicted == labels).sum()
    return batch_correct, batch_total


def main():
    batch_size = 32
    workers = 4
    gpu_id = 'cuda:2'
    dataset_path = '/home/njuciairs/rainkeeper/Projects/Datasets/crop_image'
    results_dir = '/home/njuciairs/rainkeeper/Projects/PycharmProjects/Multi-granularity-integrations3/results/'
    seed = time.time()

    cut_list = [1, 2, 3]
    for i in range(len(cut_list)):
        for j in range(cut_list[i] * cut_list[i]):
            checkpoint_save_dir = '/home/njuciairs/rainkeeper/Projects/PycharmProjects/Multi-granularity-integrations3/checkpoint0/' + str(cut_list[i] * cut_list[i]) + '_' + str(j)
            result_file_path = os.path.join(results_dir, 'result_file_' + str(cut_list[i] * cut_list[i]) + '_' + str(j) + '.txt')
            if not os.path.exists(os.path.dirname(result_file_path)):
                os.makedirs(os.path.dirname(result_file_path))
            result_file = open(result_file_path, 'a')

            max_epoch = 0
            # for filename in glob.glob('*.tar'):
            for filename in glob.glob(checkpoint_save_dir + '/*.tar'):
                # print('base name:', os.path.basename(filename))
                epoch = int(re.sub('\D', '', os.path.basename(filename)))
                # epoch = int(filename[:-18])
                # print('current filename', filename)
                # print('current epoch', epoch)
                if epoch > max_epoch:
                    max_epoch = epoch
            best_model_name = str(max_epoch) + 'checkpoint.pth.tar'
            pre_model_path = os.path.join(checkpoint_save_dir, best_model_name)
            # print('max_epoch', max_epoch)

            device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
            torch.cuda.manual_seed(seed)

            model = vggnet()
            checkpoint = torch.load(pre_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            model.eval()

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            testset = ImageDataset(image_dir=os.path.join(dataset_path, 'test_image', str(cut_list[i] * cut_list[i]) + '_' + str(j)),
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       normalize,
                                   ]),
                                   train=False)
            test_loader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=batch_size, num_workers=workers)

            correct = 0.0
            total = 0.0

            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                labels = labels.to(device)
                for s in range(labels.size(0)):
                    _, predicted = torch.max(outputs.data, 1)
                    result_file.write(str(labels[s].item()) + '\t' + str(predicted[s].item()) + '\n')

                batch_correct, batch_total = accuracy(outputs, labels)
                correct += batch_correct
                total += batch_total
            print('crop:%d, best_epoch:%d, test correct:%d, test total:%d, test_precsion:%.6f' % (j, max_epoch, correct, total, float(correct)/total))
            result_file.close()

if __name__ == '__main__':
    main()

