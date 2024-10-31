import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import numpy as np
from tqdm import tqdm

class ViTModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTModel, self).__init__()
        # 使用预训练权重
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vit_b_16(weights=weights)
        # 修改最后一层以适应新的类别数
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class AnchorNetwork(nn.Module):
    def __init__(self, num_classes=2):
        super(AnchorNetwork, self).__init__()
        self.model = ViTModel(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class CSPAgent:
    def __init__(self, class_splits, epsilon=0.05, device='cpu'):
        self.class_splits = class_splits
        self.num_classes_list = [len(classes) for classes in class_splits]
        self.epsilon = epsilon
        self.device = device

        self.anchors = []
        self.anchor_tasks = []
        self.alphas = {}
        self.task_performances = {}

    def optimize_alpha(self, data_loader):
        num_anchors = len(self.anchors)
        best_alpha = None
        best_performance = 0.0

        # 在单位 simplex 上随机采样多个 alpha
        for _ in range(100):
            alpha = np.random.dirichlet(np.ones(num_anchors))
            performance = self.evaluate_policy(data_loader, alpha)
            if performance > best_performance:
                best_performance = performance
                best_alpha = alpha

        return best_alpha, best_performance

    def evaluate_policy(self, data_loader, alpha=None):
        num_anchors = len(self.anchors)
        if alpha is None:
            # 如果未提供 alpha，使用均匀分布
            alpha = np.ones(num_anchors) / num_anchors
        else:
            # 确保 alpha 的长度与锚点数量一致
            if len(alpha) < num_anchors:
                alpha = np.pad(alpha, (0, num_anchors - len(alpha)), 'constant')

        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.get_policy(inputs, alpha)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def get_policy(self, inputs, alpha, active_anchor_indices=None):
        logits = None
        num_anchors = len(self.anchors)
        # 确保 alpha 的长度与锚点数量一致
        if len(alpha) < num_anchors:
            alpha = np.pad(alpha, (0, num_anchors - len(alpha)), 'constant')
        if active_anchor_indices is None:
            active_anchor_indices = range(num_anchors)
        for idx in active_anchor_indices:
            weight = alpha[idx]
            if weight > 0:
                anchor = self.anchors[idx]
                anchor.eval()  # 确保锚点在评估模式
                with torch.no_grad():
                    outputs = anchor(inputs)
                if logits is None:
                    logits = weight * outputs
                else:
                    logits += weight * outputs
        return logits

    def train_on_task(self, train_loader, test_loader, task_id, num_epochs=10, learning_rate=0.001, output_dir=None):
        # 添加新锚点网络
        num_classes = self.num_classes_list[task_id]
        new_anchor = AnchorNetwork(num_classes=num_classes).to(self.device)
        self.anchors.append(new_anchor)
        self.anchor_tasks.append(task_id)

        # 初始化 alpha 权重向量，只训练新锚点，权重为 1
        num_anchors = len(self.anchors)
        alpha = np.zeros(num_anchors)
        alpha[-1] = 1  # 新锚点的权重为 1

        # 定义优化器，只优化新锚点的参数
        optimizer = torch.optim.Adam(new_anchor.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        # 初始化记录训练过程的损失和准确率
        epoch_losses = []
        epoch_accuracies = []

        # 训练新锚点
        for epoch in range(num_epochs):
            new_anchor.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # 添加训练进度条
            progress_bar = tqdm(train_loader, desc=f"Task {task_id} - Epoch {epoch + 1}/{num_epochs}", unit="batch")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                # 只计算新锚点的输出
                outputs = new_anchor(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新进度条
                avg_loss = running_loss / (total / train_loader.batch_size)
                accuracy = 100 * correct / total
                progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy)

            avg_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(accuracy)
            tqdm.write(f"Task {task_id} - Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # 训练完成后，优化 alpha 权重
        best_alpha, best_performance = self.optimize_alpha(test_loader)

        # 确保 best_alpha 的长度与当前锚点数量一致
        if len(best_alpha) < num_anchors:
            best_alpha = np.pad(best_alpha, (0, num_anchors - len(best_alpha)), 'constant')

        self.alphas[task_id] = best_alpha

        # 检查性能提升是否超过阈值
        if len(self.anchors) > 1:
            # 计算不使用新锚点时的性能
            old_alpha = np.copy(best_alpha)
            old_alpha[-1] = 0  # 将新锚点的权重设为 0
            sum_alpha = old_alpha.sum()
            if sum_alpha == 0:
                # 避免除以零，将权重平均分配给其他锚点
                old_alpha[:-1] = 1.0 / (num_anchors - 1)
            else:
                old_alpha /= sum_alpha  # 重新归一化

            performance_old = self.evaluate_policy(test_loader, alpha=old_alpha)
            performance_improvement = best_performance - performance_old
            tqdm.write(f"Task {task_id}: Performance improvement with new anchor: {performance_improvement:.2f}%")

            if performance_improvement < self.epsilon:
                # 性能提升不足，移除新锚点
                self.anchors.pop()
                self.anchor_tasks.pop()
                self.alphas[task_id] = old_alpha
                tqdm.write(f"Task {task_id}: New anchor discarded due to insufficient performance improvement.")
            else:
                # 性能提升足够，保留新锚点
                tqdm.write(f"Task {task_id}: New anchor retained.")
        else:
            # 第一个锚点，直接保留
            tqdm.write(f"Task {task_id}: First anchor added.")

        # 如果指定了 output_dir，保存训练结果
        if output_dir is not None:
            try:
                # 保存训练结果到 CSV 文件
                csv_filename = os.path.join(output_dir, f'training_results_task_{task_id}.csv')
                df = pd.DataFrame({
                    'Epoch': np.arange(1, num_epochs + 1),
                    'Loss': epoch_losses,
                    'Accuracy': epoch_accuracies
                })
                df.to_csv(csv_filename, index=False)
                tqdm.write(f"CSPAgent training results saved to {csv_filename}")
            except Exception as e:
                tqdm.write(f"Error saving CSPAgent training results: {e}")

            # 绘制并保存性能曲线
            try:
                from src.plot_performance import plot_performance_single_task
                plot_performance_single_task(csv_filename, output_dir, task_id)
                tqdm.write(f"CSPAgent performance plot saved.")
            except Exception as e:
                tqdm.write(f"Error plotting CSPAgent performance: {e}")

        return new_anchor, epoch_losses, epoch_accuracies  # 返回新锚点和实际的损失、准确率列表

    def set_train_mode(self):
        for anchor in self.anchors:
            anchor.train()

    def set_eval_mode(self):
        for anchor in self.anchors:
            anchor.eval()
