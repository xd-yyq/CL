import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CIFARDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_cifar10_batch(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)  # (num_samples, 32, 32, 3)
    return data, labels

def load_cifar10_data(data_dir):
    train_data = []
    train_labels = []
    for i in range(1, 6):
        data, labels = load_cifar10_batch(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(data)
        train_labels.append(labels)

    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    test_data, test_labels = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))

    return (train_data, train_labels), (test_data, test_labels)

def load_cifar100_data(data_dir):
    with open(os.path.join(data_dir, 'train'), 'rb') as f:
        train = pickle.load(f, encoding='bytes')
    train_data = train[b'data']
    train_labels = train[b'fine_labels']
    train_data = train_data.reshape(len(train_data), 3, 32, 32).transpose(0, 2, 3, 1)

    with open(os.path.join(data_dir, 'test'), 'rb') as f:
        test = pickle.load(f, encoding='bytes')
    test_data = test[b'data']
    test_labels = test[b'fine_labels']
    test_data = test_data.reshape(len(test_data), 3, 32, 32).transpose(0, 2, 3, 1)

    return (train_data, train_labels), (test_data, test_labels)

def split_dataset_by_classes(data, labels, classes):
    labels = np.array(labels)  # 确保 labels 是一个 numpy 数组
    indices = [i for i, label in enumerate(labels) if label in classes]
    split_data = data[indices]  # 根据索引提取数据
    split_labels = labels[indices]  # 根据索引提取标签
    return split_data, split_labels

def remap_labels(labels, classes):
    label_map = {original: new for new, original in enumerate(classes)}
    remapped_labels = np.array([label_map[label] for label in labels])
    return remapped_labels

def get_task_datasets(data_dir, dataset_type, num_tasks):
    if dataset_type == 'cifar10':
        data_dir = os.path.join(data_dir, 'cifar-10-batches-py')
        (train_data, train_labels), (test_data, test_labels) = load_cifar10_data(data_dir)
        num_classes = 10
    elif dataset_type == 'cifar100':
        data_dir = os.path.join(data_dir, 'cifar-100-python')
        (train_data, train_labels), (test_data, test_labels) = load_cifar100_data(data_dir)
        num_classes = 100
    else:
        raise ValueError("Unsupported dataset type. Choose either 'cifar10' or 'cifar100'.")

    # 计算每个任务的类别数
    classes_per_task = num_classes // num_tasks
    # 创建类别划分
    class_splits = [list(range(i * classes_per_task, (i + 1) * classes_per_task)) for i in range(num_tasks)]

    # 如果有剩余的类别，分配到最后一个任务
    remaining_classes = num_classes % num_tasks
    if remaining_classes > 0:
        last_task_classes = list(range(num_tasks * classes_per_task, num_classes))
        class_splits[-1].extend(last_task_classes)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # 确保图像大小符合 ViT 的输入要求
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    task_train_datasets = []
    task_test_datasets = []
    remapped_class_splits = []

    for i, classes in enumerate(class_splits):
        # 处理训练数据
        split_train_data, split_train_labels = split_dataset_by_classes(train_data, train_labels, classes)
        remapped_train_labels = remap_labels(split_train_labels, classes)
        task_train_datasets.append(CIFARDataset(split_train_data, remapped_train_labels, transform=transform))
        print(f"Task {i} original train classes: {set(split_train_labels)} -> remapped train classes: {set(remapped_train_labels)}")

        # 处理测试数据
        split_test_data, split_test_labels = split_dataset_by_classes(test_data, test_labels, classes)
        remapped_test_labels = remap_labels(split_test_labels, classes)
        task_test_datasets.append(CIFARDataset(split_test_data, remapped_test_labels, transform=transform))
        print(f"Task {i} original test classes: {set(split_test_labels)} -> remapped test classes: {set(remapped_test_labels)}")

        remapped_class_splits.append(set(remapped_train_labels))

    return task_train_datasets, task_test_datasets, class_splits, remapped_class_splits

def check_task_datasets(task_datasets, class_splits):
    all_correct = True
    for i, dataset in enumerate(task_datasets):
        classes_in_dataset = set(dataset.labels.numpy())  # 从数据集中获取实际类
        expected_classes = set(range(len(class_splits[i])))  # 期望的类应该根据每个任务重新映射后的类数量确定
        if classes_in_dataset != expected_classes:
            all_correct = False
            raise AssertionError(f"Task {i} contains incorrect classes: {classes_in_dataset} vs {expected_classes}")
    if all_correct:
        print("All tasks contain correct classes.")
