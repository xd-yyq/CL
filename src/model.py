# src/model.py

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
    def __init__(self, class_splits, epsilon=0.05, device='cpu', model_dir='models'):
        self.class_splits = class_splits
        self.num_classes_list = [len(classes) for classes in class_splits]
        self.epsilon = epsilon
        self.device = device

        # 修改：将 anchors 从模型对象列表改为模型路径列表
        self.anchors = []
        self.anchor_tasks = []
        self.alphas = {}
        self.task_performances = {}
        self.model_dir = model_dir  # 模型保存的目录
        os.makedirs(self.model_dir, exist_ok=True)

    def save_anchor(self, anchor, task_id):
        model_path = os.path.join(self.model_dir, f'anchor_task_{task_id}.pth')
        torch.save(anchor.state_dict(), model_path)
        return model_path

    def load_anchor(self, task_id, num_classes):
        model_path = os.path.join(self.model_dir, f'anchor_task_{task_id}.pth')
        anchor = AnchorNetwork(num_classes=num_classes)
        anchor.load_state_dict(torch.load(model_path, map_location=self.device))
        return anchor

    def optimize_alpha_grad(self, test_loader, num_steps=50, learning_rate=0.01):
        """
        基于梯度的方法优化 alpha 权重。
        """
        num_anchors = len(self.anchors)
        if num_anchors == 0:
            return None, 0.0

        # 初始化 alpha 参数，使用 softmax 确保在 simplex 上
        alpha_param = torch.nn.Parameter(torch.ones(num_anchors, device=self.device) / num_anchors)
        optimizer = torch.optim.Adam([alpha_param], lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # 使用 test_loader 的第一个批次作为验证集
        try:
            inputs, labels = next(iter(test_loader))
        except StopIteration:
            print("Test loader is empty.")
            return None, 0.0

        inputs, labels = inputs.to(self.device), labels.to(self.device)

        for step in range(num_steps):
            optimizer.zero_grad()
            alpha = torch.softmax(alpha_param, dim=0)
            logits = self.get_policy(inputs, alpha=alpha)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # 限制 alpha_param 的范围以防数值问题
            with torch.no_grad():
                alpha_param.clamp_(-5, 5)

            if step % 10 == 0 or step == num_steps -1:
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=1)
                    accuracy = (predictions == labels).float().mean().item()
                print(f"Alpha Optimization Step {step+1}/{num_steps}, Loss: {loss.item():.4f}, Accuracy: {accuracy*100:.2f}%")

        # 保持 alpha 为张量而不是 NumPy 数组
        optimized_alpha = torch.softmax(alpha_param, dim=0).detach().cpu()
        # 计算优化后的性能
        performance = self.evaluate_policy(test_loader, alpha=optimized_alpha)
        return optimized_alpha, performance

    def evaluate_policy(self, data_loader, alpha=None):
        """
        评估当前策略的准确率。如果未提供 alpha，则使用均匀分布。
        """
        num_anchors = len(self.anchors)
        if num_anchors == 0:
            return 0.0

        # 确保 alpha 在正确的设备上
        if alpha is None:
            alpha = torch.ones(num_anchors, device=self.device) / num_anchors
        else:
            # 确保 alpha 在正确的设备上
            alpha = alpha.to(self.device)

            # 确保 alpha 的长度与锚点数量一致
            if alpha.numel() < num_anchors:
                padding = torch.zeros(num_anchors - alpha.numel(), device=self.device)
                alpha = torch.cat([alpha, padding], dim=0)

        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.get_policy(inputs, alpha=alpha)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def get_policy(self, inputs, alpha, active_anchor_indices=None):
        """
        根据当前的 alpha 权重计算输出 logits。
        """
        logits = None
        num_anchors = len(self.anchors)
        # 确保 alpha 的长度与锚点数量一致
        if alpha.numel() < num_anchors:
            padding = torch.zeros(num_anchors - alpha.numel(), device=self.device)
            alpha = torch.cat([alpha, padding], dim=0)
        if active_anchor_indices is None:
            active_anchor_indices = range(num_anchors)
        for idx in active_anchor_indices:
            weight = alpha[idx]
            if weight > 0:
                task_id = self.anchor_tasks[idx]
                num_classes = self.num_classes_list[task_id]
                # 加载模型
                anchor = self.load_anchor(task_id, num_classes).to(self.device)
                anchor.eval()
                with torch.no_grad():
                    outputs = anchor(inputs)
                # 使用完后删除模型，释放显存
                del anchor
                torch.cuda.empty_cache()
                if logits is None:
                    logits = weight * outputs
                else:
                    logits += weight * outputs
        return logits

    def train_on_task(self, train_loader, test_loader, task_id, num_epochs=10, learning_rate=0.001, output_dir=None):
        """
        在给定的任务上训练 CSPAgent。
        """
        # 添加新锚点网络
        num_classes = self.num_classes_list[task_id]
        new_anchor = AnchorNetwork(num_classes=num_classes).to(self.device)
        self.anchor_tasks.append(task_id)

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

        # 训练完成后，保存新锚点模型并释放显存
        model_path = self.save_anchor(new_anchor, task_id)
        tqdm.write(f"Saved new anchor model for Task {task_id} at {model_path}")
        del new_anchor
        torch.cuda.empty_cache()
        # 将模型路径添加到 anchors 列表
        self.anchors.append(model_path)

        # 优化 alpha 权重
        optimized_alpha, best_performance = self.optimize_alpha_grad(test_loader)

        if optimized_alpha is None:
            # 如果优化失败，保留原来的 alpha
            num_anchors = len(self.anchors)
            alpha = torch.zeros(num_anchors)
            alpha[-1] = 1
            optimized_alpha = alpha
            best_performance = self.evaluate_policy(test_loader, alpha=alpha)

        self.alphas[task_id] = optimized_alpha

        # 检查性能提升是否超过阈值
        if len(self.anchors) > 1:
            # 计算不使用新锚点时的性能
            old_alpha = optimized_alpha.clone()
            old_alpha[-1] = 0  # 将新锚点的权重设为 0
            sum_alpha = old_alpha.sum()
            if sum_alpha == 0:
                # 避免除以零，将权重平均分配给其他锚点
                old_alpha[:-1] = 1.0 / (len(self.anchors) - 1)
            else:
                old_alpha /= sum_alpha  # 重新归一化

            performance_old = self.evaluate_policy(test_loader, alpha=old_alpha)
            performance_improvement = best_performance - performance_old
            tqdm.write(f"Task {task_id}: Performance improvement with new anchor: {performance_improvement:.2f}%")

            if performance_improvement < self.epsilon:
                # 性能提升不足，移除新锚点
                os.remove(self.anchors[-1])  # 删除模型文件
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

        return None, epoch_losses, epoch_accuracies  # 由于模型已保存，返回 None

    def set_train_mode(self):
        pass  # 不再需要，因为模型不在内存中

    def set_eval_mode(self):
        pass  # 不再需要，因为模型不在内存中
