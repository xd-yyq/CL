# evaluate_saved_models.py

import os
import torch
from torch.utils.data import DataLoader
from src.model import CSPAgent, AnchorNetwork
from src.data_processing import get_task_datasets
from src.config import Config
import numpy as np
from tqdm import tqdm

def main():
    # 配置参数
    config = Config()
    num_tasks = 5  # 你想评估的任务数量，这里是任务0到任务3
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 设置实验目录（请根据你的实际路径修改）
    exp_dir = os.path.join(config.OUTPUT_DIR, config.DATASET_TYPE, 'exp3')
    print(f"Experiment directory: {exp_dir}")

    # 加载数据集
    print("Loading datasets...")
    task_train_datasets, task_test_datasets, original_class_splits, _ = get_task_datasets(
        config.DATA_DIR,
        config.DATASET_TYPE,
        num_tasks
    )

    # 创建测试数据加载器
    test_loaders = [
        DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        for dataset in task_test_datasets
    ]

    # 初始化 CSP Agent
    csp_agent = CSPAgent(
        class_splits=original_class_splits,
        epsilon=0.05,
        device=device
    )

    # 遍历每个任务，加载模型并评估
    for task_id in range(num_tasks):
        print(f"\nEvaluating Task {task_id}")
        task_output_dir = os.path.join(exp_dir, f"task_{task_id}")
        csp_agent_state_path = os.path.join(task_output_dir, f'csp_agent_task_{task_id}.pth')

        if os.path.exists(csp_agent_state_path):
            # 加载 CSPAgent 的状态
            state = torch.load(csp_agent_state_path, map_location=device)
            csp_agent.anchors = []
            for anchor_state in state['anchors']:
                # 获取锚点的类别数
                num_classes = anchor_state['model.model.heads.head.weight'].size(0)
                # 重新创建 AnchorNetwork 并加载状态
                anchor = AnchorNetwork(num_classes=num_classes).to(device)
                anchor.load_state_dict(anchor_state)
                anchor.eval()
                csp_agent.anchors.append(anchor)
            csp_agent.alphas = state['alphas']

            # 评估当前任务的性能
            alpha = csp_agent.alphas.get(task_id)
            if alpha is not None:
                # 确保 alpha 是 PyTorch 张量并在正确的设备上
                if isinstance(alpha, np.ndarray) or isinstance(alpha, list):
                    alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
                else:
                    alpha = alpha.to(device)

                # 修改此处，使用正确的参数名称或位置参数
                accuracy = csp_agent.evaluate_policy(test_loaders[task_id], alpha)
                print(f"Final Evaluation on Task {task_id} - Accuracy: {accuracy:.2f}%")
            else:
                print(f"No alpha vector found for Task {task_id}")
        else:
            print(f"No saved model found for Task {task_id} at {csp_agent_state_path}")

if __name__ == "__main__":
    main()
