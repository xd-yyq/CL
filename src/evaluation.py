import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model import ViTModel, CSPAgent
from src.data_processing import get_task_datasets

def evaluate_saved_models(exp_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = './data/cifar-100-python'  # 根据您的配置进行调整
    num_tasks = 10  # 根据您的配置进行调整

    # 获取任务数据集
    task_train_datasets, task_test_datasets, original_class_splits, remapped_class_splits = get_task_datasets(
        data_dir)

    test_loaders = [DataLoader(dataset, batch_size=32, shuffle=False) for dataset in task_test_datasets]

    # 初始化 CSPAgent
    csp_agent = CSPAgent(class_splits=original_class_splits, epsilon=0.05, device=device)

    # 加载保存的模型状态
    checkpoint = torch.load(os.path.join(exp_dir, f'csp_agent_task_{num_tasks - 1}.pth'), map_location=device)
    saved_anchors = checkpoint['anchors']
    saved_alphas = checkpoint['alphas']

    # 恢复锚点模型
    for anchor_state in saved_anchors:
        # 获取锚点的类别数，可以根据实际情况进行调整
        num_classes = len(csp_agent.class_splits[0])  # 假设所有锚点的类别数相同
        anchor = ViTModel(num_classes=num_classes).to(device)
        anchor.load_state_dict(anchor_state)
        csp_agent.anchors.append(anchor)

    csp_agent.alphas = saved_alphas

    # 打开评估结果文件
    eval_log_filename = os.path.join(exp_dir, 'evaluation_results.txt')
    with open(eval_log_filename, 'w', encoding='utf-8') as eval_log_file:
        # 在所有任务上评估
        for task_id in range(num_tasks):
            alpha = csp_agent.alphas.get(task_id, None)
            if alpha is None:
                # 如果没有对应的 alpha，使用均匀权重
                alpha = np.ones(len(csp_agent.anchors)) / len(csp_agent.anchors)
            accuracy = csp_agent.evaluate_policy(test_loaders[task_id], alpha=alpha)
            result_str = f"Final Evaluation on Task {task_id} - Accuracy: {accuracy:.2f}%"
            print(result_str)
            eval_log_file.write(result_str + '\n')

        # 计算平均性能
        total_performance = 0.0
        for task_id in range(num_tasks):
            alpha = csp_agent.alphas.get(task_id, None)
            if alpha is None:
                alpha = np.ones(len(csp_agent.anchors)) / len(csp_agent.anchors)
            accuracy = csp_agent.evaluate_policy(test_loaders[task_id], alpha=alpha)
            total_performance += accuracy
        average_performance = total_performance / num_tasks
        avg_perf_str = f"\nAverage Performance: {average_performance:.2f}%"
        print(avg_perf_str)
        eval_log_file.write(avg_perf_str + '\n')

        # 可选：保存 Alpha 向量
        eval_log_file.write("\nAlpha vectors for each task:\n")
        for task_id, alpha in csp_agent.alphas.items():
            alpha_str = np.array2string(alpha, precision=8, separator=', ')
            alpha_log_str = f"Task {task_id}: alpha = {alpha_str}"
            print(alpha_log_str)
            eval_log_file.write(alpha_log_str + '\n')

