import os
import torch
from torch.utils.data import DataLoader
from src.model import ViTModel, CSPAgent
from src.data_processing import get_task_datasets, check_task_datasets
from src.visualization import visualize_task_anchor_relationship
from src.training import train_model, evaluate_model, save_model, load_model
from src.plot_performance import plot_performance
from src.config import Config
import numpy as np
import hashlib
import json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Logger:
    def __init__(self, output_dir):
        self.log_filename = os.path.join(output_dir, 'results.txt')
        self.log_file = open(self.log_filename, 'w', encoding='utf-8')

    def log(self, message):
        print(message)
        self.log_file.write(message + '\n')

    def close(self):
        self.log_file.close()


def get_next_experiment_dir(dataset_output_dir):
    # 查找以 'exp' 开头的现有目录
    existing_exp_dirs = [d for d in os.listdir(dataset_output_dir)
                         if os.path.isdir(os.path.join(dataset_output_dir, d)) and d.startswith('exp')]
    # 提取数字部分
    exp_numbers = [int(d[3:]) for d in existing_exp_dirs if d[3:].isdigit()]
    if exp_numbers:
        next_exp_number = max(exp_numbers) + 1
    else:
        next_exp_number = 1
    # 创建新的实验目录
    exp_dir = os.path.join(dataset_output_dir, f'exp{next_exp_number}')
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def main():
    # 使用配置文件中的数据集类型
    config = Config()
    dataset_type = config.DATASET_TYPE

    # 创建数据集输出目录（cifar10、cifar100 等）
    dataset_output_dir = os.path.join(config.OUTPUT_DIR, dataset_type)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # 创建新的实验目录
    exp_dir = get_next_experiment_dir(dataset_output_dir)
    # 使用新的实验目录初始化 Logger
    logger = Logger(exp_dir)
    logger.log(f'Using device: {config.DEVICE}')
    logger.log(f'Using dataset: {dataset_type}')
    logger.log(f'Number of tasks: {config.NUM_TASKS}')

    # 获取任务数据集，基于 Config.DATASET_TYPE 选择 cifar10 或 cifar100
    try:
        task_train_datasets, task_test_datasets, original_class_splits, remapped_class_splits = get_task_datasets(
            config.DATA_DIR, dataset_type, config.NUM_TASKS)
        logger.log("Successfully loaded task datasets.")
    except Exception as e:
        logger.log(f"Error loading task datasets: {e}")
        logger.close()
        return

    try:
        check_task_datasets(task_train_datasets, original_class_splits)
        check_task_datasets(task_test_datasets, original_class_splits)
        logger.log("Task datasets passed consistency checks.")
    except Exception as e:
        logger.log(f"Task datasets consistency check failed: {e}")
        logger.close()
        return

    train_loaders = [
        DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        for dataset in task_train_datasets
    ]
    test_loaders = [
        DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        for dataset in task_test_datasets
    ]

    # 生成类别划分的字符串和哈希值
    class_splits_str = json.dumps(original_class_splits, sort_keys=True)
    class_splits_hash = hashlib.md5(class_splits_str.encode('utf-8')).hexdigest()

    # 更新实验名称，包含类别划分的哈希值
    experiment_name = f"{dataset_type}_tasks_{config.NUM_TASKS}_hash_{class_splits_hash}"
    logger.log(f'Experiment name: {experiment_name}')

    # 获取项目根目录（上一级目录）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 定义 checkpoint 目录
    checkpoint_dir = os.path.join(project_root, 'checkpoint')
    checkpoint_dir = os.path.abspath(os.path.normpath(checkpoint_dir))
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 在 checkpoint 目录下创建一个子目录，包含实验名称
    experiment_checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
    experiment_checkpoint_dir = os.path.abspath(os.path.normpath(experiment_checkpoint_dir))
    os.makedirs(experiment_checkpoint_dir, exist_ok=True)

    # 打印调试信息，检查路径是否正确
    logger.log(f'Checkpoint directory: {checkpoint_dir}')
    logger.log(f'Experiment checkpoint directory: {experiment_checkpoint_dir}')

    # 保存实验配置到 JSON 文件
    config_path = os.path.join(experiment_checkpoint_dir, 'config.json')
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'DATASET_TYPE': dataset_type,
                'NUM_TASKS': config.NUM_TASKS,
                'CLASS_SPLITS': original_class_splits
            }, f, indent=4)
        logger.log(f"Experiment configuration saved to {config_path}")
    except Exception as e:
        logger.log(f"Error saving experiment configuration: {e}")
        logger.close()
        return

    # 初始化 CSPAgent
    try:
        csp_agent = CSPAgent(class_splits=original_class_splits, epsilon=0.05, device=config.DEVICE)
        logger.log("CSPAgent initialized successfully.")
    except Exception as e:
        logger.log(f"Error initializing CSPAgent: {e}")
        logger.close()
        return

    # 创建一个字典，用于保存所有任务的训练结果
    training_results = {}

    # 依次训练每个任务
    for task_id in range(config.NUM_TASKS):
        logger.log(f"\nTraining on Task {task_id}")

        # 动态获取当前任务的类别数
        num_classes = len(original_class_splits[task_id])
        logger.log(f"Task {task_id} has {num_classes} classes.")

        # 生成模型文件名，包含任务编号和类别数
        model_filename = f"task_{task_id}_classes_{num_classes}_model.pt"

        # 生成模型文件路径
        model_filepath = os.path.join(experiment_checkpoint_dir, model_filename)
        model_filepath = os.path.abspath(os.path.normpath(model_filepath))

        # 确保模型文件的目录存在
        os.makedirs(os.path.dirname(model_filepath), exist_ok=True)

        # 定义任务的主目录
        task_output_dir = os.path.join(exp_dir, f"task_{task_id}")
        os.makedirs(task_output_dir, exist_ok=True)

        # 在日志中打印模型文件路径
        logger.log(f"Looking for standalone model at {model_filepath}")

        # 检查模型是否已存在
        if os.path.exists(model_filepath):
            logger.log(f"Found existing standalone model for Task {task_id} at {model_filepath}. Loading model.")
            try:
                # 加载模型
                standalone_model = ViTModel(num_classes=num_classes).to(config.DEVICE)
                standalone_model.load_state_dict(torch.load(model_filepath, map_location=config.DEVICE))
                standalone_model.eval()
                logger.log(f"Standalone model for Task {task_id} loaded successfully.")
                # 评估初始性能
                initial_performance = evaluate_model(standalone_model, test_loaders[task_id], config.DEVICE, task_id)
                logger.log(f"Task {task_id} - Initial Performance: {initial_performance:.2f}%")
            except Exception as e:
                logger.log(f"Error loading or evaluating standalone model for Task {task_id}: {e}")
                logger.close()
                return
        else:
            logger.log(f"No existing standalone model for Task {task_id} at {model_filepath}. Starting Standalone Training.")
            # 如果未找到模型，需要进行 Standalone Training
            try:
                # 初始化模型
                standalone_model = ViTModel(num_classes=num_classes).to(config.DEVICE)
                # 定义损失函数和优化器
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(standalone_model.parameters(), lr=config.LEARNING_RATE)
                # 训练模型
                standalone_model, epoch_losses, epoch_accuracies = train_model(
                    standalone_model,
                    train_loaders[task_id],
                    criterion,
                    optimizer,
                    config.NUM_EPOCHS,
                    config.DEVICE,
                    task_id,
                    task_output_dir  # 保存到任务目录
                )
                # 保存模型
                save_model(standalone_model, model_filepath)
                logger.log(f"Standalone model for Task {task_id} saved to {model_filepath}")
                # 评估初始性能
                initial_performance = evaluate_model(standalone_model, test_loaders[task_id], config.DEVICE, task_id)
                logger.log(f"Task {task_id} - Initial Performance after Standalone Training: {initial_performance:.2f}%")
            except Exception as e:
                logger.log(f"Error during Standalone Training for Task {task_id}: {e}")
                logger.close()
                return

        # 将 initial_performance 传递给 csp_agent
        csp_agent.task_performances[task_id] = {
            'initial_performance': initial_performance
        }

        # 在 CSPAgent 中训练任务
        try:
            # 调用 CSPAgent 的 train_on_task 方法，传递 task_output_dir
            model_cspagent, csp_epoch_losses, csp_epoch_accuracies = csp_agent.train_on_task(
                train_loader=train_loaders[task_id],
                test_loader=test_loaders[task_id],
                task_id=task_id,
                num_epochs=config.NUM_EPOCHS,
                learning_rate=config.LEARNING_RATE,
                output_dir=task_output_dir  # 保存到 task_output_dir
            )
            logger.log(f"Task {task_id} - CSPAgent training completed.")
        except Exception as e:
            logger.log(f"Error training CSPAgent on Task {task_id}: {e}")
            logger.close()
            return

        # 保存 CSPAgent 的状态
        csp_agent_state_path = os.path.join(task_output_dir, f'csp_agent_task_{task_id}.pth')
        csp_agent_state_path = os.path.abspath(os.path.normpath(csp_agent_state_path))
        try:
            torch.save({
                'anchors': [anchor.state_dict() for anchor in csp_agent.anchors],
                'alphas': csp_agent.alphas
            }, csp_agent_state_path)
            logger.log(f"CSPAgent state saved to {csp_agent_state_path}")
        except Exception as e:
            logger.log(f"Error saving CSPAgent state for Task {task_id}: {e}")
            logger.close()
            return

        # 在当前任务上评估
        alpha = csp_agent.alphas.get(task_id)
        if alpha is not None:
            try:
                accuracy = csp_agent.evaluate_policy(test_loaders[task_id], alpha=alpha)
                logger.log(f"Task {task_id} - Accuracy after CSPAgent training: {accuracy:.2f}%")
            except Exception as e:
                logger.log(f"Error evaluating policy for Task {task_id}: {e}")
                logger.close()
                return
        else:
            logger.log(f"No Alpha vector found for Task {task_id}. Skipping evaluation.")

    # 在所有任务上评估，检查是否发生遗忘
    for task_id in range(config.NUM_TASKS):
        alpha = csp_agent.alphas.get(task_id)
        if alpha is not None:
            try:
                accuracy = csp_agent.evaluate_policy(test_loaders[task_id], alpha=alpha)
                logger.log(f"Final Evaluation on Task {task_id} - Accuracy: {accuracy:.2f}%")
                csp_agent.task_performances[task_id]['final_performance'] = accuracy
            except Exception as e:
                logger.log(f"Error during final evaluation on Task {task_id}: {e}")
                logger.close()
                return
        else:
            logger.log(f"No Alpha vector found for Task {task_id}. Skipping final evaluation.")

    # 计算平均性能
    total_performance = sum(
        csp_agent.task_performances[task_id]['final_performance']
        for task_id in csp_agent.task_performances if
        'final_performance' in csp_agent.task_performances[task_id]
    )
    num_evaluated_tasks = len([
        task_id for task_id in csp_agent.task_performances if
        'final_performance' in csp_agent.task_performances[task_id]
    ])
    if num_evaluated_tasks > 0:
        average_performance = total_performance / num_evaluated_tasks
    else:
        average_performance = 0.0
    logger.log(f"\nAverage Performance: {average_performance:.2f}%")

    # 计算平均遗忘
    total_forgetting = sum(
        csp_agent.task_performances[task_id]['initial_performance'] -
        csp_agent.task_performances[task_id]['final_performance']
        for task_id in csp_agent.task_performances
        if 'final_performance' in csp_agent.task_performances[task_id]
    )
    if num_evaluated_tasks > 0:
        average_forgetting = total_forgetting / num_evaluated_tasks
    else:
        average_forgetting = 0.0
    logger.log(f"Average Forgetting: {average_forgetting:.2f}%")

    # 计算模型大小
    if len(csp_agent.anchors) > 0:
        single_anchor_params = sum(p.numel() for p in csp_agent.anchors[0].parameters())
        total_params = sum(p.numel() for anchor in csp_agent.anchors for p in anchor.parameters())
        model_size = total_params / single_anchor_params
        logger.log(f"Model Size (relative to single anchor): {model_size:.2f}")
    else:
        logger.log("No anchors found. Cannot compute model size.")

    # 计算平均前向迁移
    total_forward_transfer = 0.0
    count_forward_transfer = 0
    for task_id in range(1, config.NUM_TASKS):
        prev_alpha = csp_agent.alphas.get(task_id - 1)
        if prev_alpha is not None:
            try:
                forward_performance = csp_agent.evaluate_policy(test_loaders[task_id], alpha=prev_alpha)
                initial_performance = csp_agent.task_performances[task_id]['initial_performance']
                forward_transfer = forward_performance - initial_performance
                total_forward_transfer += forward_transfer
                count_forward_transfer += 1
            except Exception as e:
                logger.log(f"Error calculating forward transfer for Task {task_id}: {e}")
                logger.close()
                return
    if count_forward_transfer > 0:
        average_forward_transfer = total_forward_transfer / count_forward_transfer
    else:
        average_forward_transfer = 0.0
    logger.log(f"Average Forward Transfer: {average_forward_transfer:.2f}%")

    # 总结锚点和任务
    logger.log("\nSummary of Anchors and Tasks:")
    logger.log("Anchors (tasks that became anchors):")
    for idx, anchor_task_id in enumerate(csp_agent.anchor_tasks):
        logger.log(f"Anchor {idx}: Task {anchor_task_id}")

    logger.log("\nTasks represented as convex combinations:")
    non_anchor_tasks = set(range(config.NUM_TASKS)) - set(csp_agent.anchor_tasks)
    if non_anchor_tasks:
        for task_id in non_anchor_tasks:
            logger.log(f"Task {task_id}: represented by convex combination of anchors")
    else:
        logger.log("None")

    logger.log("\nAlpha vectors for each task:")
    for task_id, alpha in csp_agent.alphas.items():
        alpha_str = np.array2string(alpha, precision=8, separator=', ')
        logger.log(f"Task {task_id}: alpha = {alpha_str}")

    # 绘制任务与锚点的关系图
    try:
        visualize_task_anchor_relationship(csp_agent, config.NUM_TASKS, exp_dir)
        logger.log("Visualized task-anchor relationships successfully.")
    except Exception as e:
        logger.log(f"Error visualizing task-anchor relationships: {e}")

    logger.log("All tasks completed.")

    # 调用绘图函数，绘制性能曲线并保存
    try:
        plot_performance(exp_dir, config.NUM_TASKS, config.NUM_EPOCHS)
        logger.log("Performance curves plotted successfully.")
    except Exception as e:
        logger.log(f"Error plotting performance curves: {e}")

    # 关闭日志文件
    logger.close()


if __name__ == '__main__':
    main()
