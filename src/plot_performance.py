import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_performance(output_dir, num_tasks, num_epochs):
    for task_id in range(num_tasks):
        task_output_dir = os.path.join(output_dir, f'task_{task_id}')
        csv_filename = os.path.join(task_output_dir, f'training_results_task_{task_id}.csv')
        if not os.path.exists(csv_filename):
            print(f"File {csv_filename} not found.")
            continue

        df = pd.read_csv(csv_filename)
        epochs = df['Epoch']
        losses = df['Loss']
        accuracies = df['Accuracy']

        # 绘制损失曲线
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, label=f'Task {task_id} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Task {task_id} Training Loss')
        plt.legend()

        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracies, label=f'Task {task_id} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Task {task_id} Training Accuracy')
        plt.legend()

        plt.tight_layout()
        # 保存图像到任务特定的子目录
        plt.savefig(os.path.join(task_output_dir, f'performance_task_{task_id}.png'))
        plt.close()

def plot_performance_single_task(csv_filename, output_dir, task_id):
    df = pd.read_csv(csv_filename)
    epochs = df['Epoch']
    losses = df['Loss']
    accuracies = df['Accuracy']

    # 绘制损失曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label=f'Task {task_id} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Task {task_id} Training Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label=f'Task {task_id} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Task {task_id} Training Accuracy')
    plt.legend()

    plt.tight_layout()
    # 保存图像到任务特定的子目录
    plt.savefig(os.path.join(output_dir, f'performance_task_{task_id}.png'))
    plt.close()
