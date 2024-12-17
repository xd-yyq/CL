import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.plot_performance import plot_performance_single_task  # 确保正确导入

def train_model(model, train_loader, criterion, optimizer, num_epochs, device, task_id, output_dir):
    model.to(device)
    epoch_losses = []
    epoch_accuracies = []

    # 为每个训练过程创建子目录
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f'training_results_task_{task_id}.csv')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Task {task_id} - Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)  # 修正损失计算

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条信息
            avg_loss = running_loss / total
            accuracy = 100 * correct / total
            progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy)

        avg_loss = running_loss / total
        accuracy = 100 * correct / total
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)
        tqdm.write(f"Task {task_id} - Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # 将每个 epoch 的结果保存到 CSV 文件
        df = pd.DataFrame({
            'Epoch': np.arange(1, len(epoch_losses) + 1),
            'Loss': epoch_losses,
            'Accuracy': epoch_accuracies
        })
        try:
            df.to_csv(csv_filename, index=False)
            print(f"[INFO] Task {task_id}: Saved training results to {csv_filename}")
        except Exception as e:
            print(f"[ERROR] Task {task_id}: Failed to save training results to CSV: {e}")
            raise e

    # 在训练结束后调用绘图函数
    try:
        plot_performance_single_task(csv_filename, output_dir, task_id)
        print(f"[INFO] Task {task_id}: Performance plot saved successfully.")
    except Exception as e:
        print(f"[ERROR] Task {task_id}: Failed to plot performance: {e}")

    # 返回模型对象、损失和准确率
    return model, epoch_losses, epoch_accuracies

def evaluate_model(model, test_loader, device, task_id):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f"Evaluating Task {task_id}", unit="batch")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    tqdm.write(f"Task {task_id} - Accuracy: {accuracy:.2f}%")

    # 评估完成后，将模型移动到 CPU 并释放显存
    model.to('cpu')
    torch.cuda.empty_cache()

    return accuracy


def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
