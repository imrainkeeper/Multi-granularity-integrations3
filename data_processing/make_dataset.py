import csv
import os
import cv2
import math
import sys

crop_image = lambda img, x0, y0, crop_w, crop_h: img[y0:y0 + crop_h, x0:x0 + crop_w]

def segmentation(cut_list, csv_file_path, dataset_path):
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        image_paths_csv_reader = csv.reader(csv_file)
        for index, path in enumerate(image_paths_csv_reader):
            print('processing %d-th image' % index)
            image_path = os.path.join(dataset_path, path[0])
            image_short_name = os.path.basename(image_path)[:-4]
            image = cv2.imread(image_path, 0)
            height, width = image.shape

            for i in range(0, len(cut_list)):
                # crop_csv_file_path = csv_file_path.replace('MURA-v1.1', 'MURA-v1.1-' + str(cut_list[i] * cut_list[i]))
                # crop_csv_file_dir = os.path.dirname(crop_csv_file_path)
                crop_csv_file_dir = os.path.dirname(csv_file_path.replace('MURA-v1.1', 'MURA-v1.1-' + str(cut_list[i] * cut_list[i])))
                if not os.path.exists(crop_csv_file_dir):
                    os.makedirs(crop_csv_file_dir)
                for j in range(0, cut_list[i] * cut_list[i]):
                    csv_file_name = os.path.basename(csv_file_path)
                    crop_csv_file_path = os.path.join(crop_csv_file_dir, csv_file_name[:-4] + str(j) + '.csv')
                    with open(crop_csv_file_path, 'a', encoding='utf-8', newline='') as crop_csv_file:
                        crop_image_paths_csv_writer = csv.writer(crop_csv_file, dialect='excel')
                        crop_w = round(math.floor(width / cut_list[i]))
                        crop_h = round(math.floor(height / cut_list[i]))
                        x0 = crop_w * (j % cut_list[i])
                        y0 = crop_h * round(math.floor(j / cut_list[i]))
                        output_image_path = image_path.replace('MURA-v1.1',
                                                               'MURA-v1.1-' + str(cut_list[i] * cut_list[i])).replace(
                            image_short_name, image_short_name + '_' + str(j))
                        output_image_dir = os.path.dirname(output_image_path)
                        if not os.path.exists(output_image_dir):
                            os.makedirs(output_image_dir)
                        cv2.imwrite(output_image_path, crop_image(image, x0, y0, crop_w, crop_h))
                        crop_image_path_short = path[0].replace('MURA-v1.1',
                                                                'MURA-v1.1-' + str(cut_list[i] * cut_list[i])).replace(
                            image_short_name, image_short_name + '_' + str(j))
                        crop_image_paths_csv_writer.writerow([crop_image_path_short])


def segmentation_image(cut_list, train_image_list, test_image_list, input_image_dir, output_image_dir):
    for index in range(len(train_image_list)):
        print('processing %d-th image in train image list' % index)
        train_image_name = train_image_list[index]
        train_image_path = os.path.join(input_image_dir, train_image_name)
        train_image = cv2.imread(train_image_path, 0)
        height, width = train_image.shape

        for i in range(0, len(cut_list)):
            for j in range(0, cut_list[i] * cut_list[i]):
                current_output_train_image_dir = os.path.join(output_image_dir, 'train_image', str(cut_list[i] * cut_list[i]) + '_' + str(j))
                if not os.path.exists(current_output_train_image_dir):
                    os.makedirs(current_output_train_image_dir)
                crop_w = round(math.floor(width / cut_list[i]))
                crop_h = round(math.floor(height / cut_list[i]))
                x0 = crop_w * (j % cut_list[i])
                y0 = crop_h * round(math.floor(j / cut_list[i]))
                output_image = crop_image(train_image, x0, y0, crop_w, crop_h)
                output_resized_image = cv2.resize(output_image, (224, 224), cv2.INTER_CUBIC)
                output_train_image_path = os.path.join(current_output_train_image_dir, train_image_name)
                cv2.imwrite(output_train_image_path, output_resized_image)

    for index in range(len(test_image_list)):
        print('processing %d-th image in test image list' % index)
        test_image_name = test_image_list[index]
        test_image_path = os.path.join(input_image_dir, test_image_name)
        test_image = cv2.imread(test_image_path, 0)
        height, width = test_image.shape

        for i in range(0, len(cut_list)):
            for j in range(0, cut_list[i] * cut_list[i]):
                current_output_test_image_dir = os.path.join(output_image_dir, 'test_image', str(cut_list[i] * cut_list[i]) + '_' + str(j))
                if not os.path.exists(current_output_test_image_dir):
                    os.makedirs(current_output_test_image_dir)
                crop_w = round(math.floor(width / cut_list[i]))
                crop_h = round(math.floor(height / cut_list[i]))
                x0 = crop_w * (j % cut_list[i])
                y0 = crop_h * round(math.floor(j / cut_list[i]))
                output_image = crop_image(test_image, x0, y0, crop_w, crop_h)
                output_resized_image = cv2.resize(output_image, (224, 224), cv2.INTER_CUBIC)
                output_test_image_path = os.path.join(current_output_test_image_dir, test_image_name)
                cv2.imwrite(output_test_image_path, output_resized_image)


def train_test_data(image_dir):
    normal_image_list = []
    abnormal_image_list = []
    for dir, subdirs, subfiles in os.walk(image_dir):
        for file in subfiles:
            if os.path.splitext(file)[1] == '.png':
                if os.path.splitext(file)[0][-1] == '0':
                    normal_image_list.append(file)
                elif os.path.splitext(file)[0][-1] == '1':
                    abnormal_image_list.append(file)
                else:
                    print('error label')
                    sys.exit(0)
    print('normal image num:', len(normal_image_list))
    print('abnormal image num:', len(abnormal_image_list))
    print('normal image list', normal_image_list)
    print('abnormal image list', abnormal_image_list)
    train_image_list = []
    test_image_list = []
    for i in range(len(normal_image_list)):
        if i < 0.7 * len(normal_image_list):
            train_image_list.append(normal_image_list[i])
        else:
            test_image_list.append(normal_image_list[i])
    for i in range(len(abnormal_image_list)):
        if i < 0.7 * len(abnormal_image_list):
            train_image_list.append(abnormal_image_list[i])
        else:
            test_image_list.append(abnormal_image_list[i])
    print('train image num:', len(train_image_list))
    print('test image num:', len(test_image_list))
    print('train image list', train_image_list)
    print('test image list', test_image_list)
    return train_image_list, test_image_list


input_image_dir = '/home/rainkeeper/Projects/Datasets/CXR_png'
train_image_list, test_image_list = train_test_data(input_image_dir)

output_image_dir = '/home/rainkeeper/Projects/Datasets/crop_image/'
cut_list = [1, 2, 3]
segmentation_image(cut_list, train_image_list, test_image_list, input_image_dir, output_image_dir)


# cut_list = [1, 2, 3]
#
# segmentation(cut_list, train_csv_file_path, dataset_path)
# segmentation(cut_list, test_csv_file_path, dataset_path)






