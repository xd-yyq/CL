import os
import networkx as nx
import matplotlib.pyplot as plt

def visualize_task_anchor_relationship(csp_agent, num_tasks, output_dir):
    G = nx.DiGraph()

    # 添加锚点节点
    for idx, anchor_task_id in enumerate(csp_agent.anchor_tasks):
        node_label = f"Anchor {idx}\n(Task {anchor_task_id})"
        G.add_node(node_label, color='red')

    # 添加非锚点任务节点
    non_anchor_tasks = set(range(num_tasks)) - set(csp_agent.anchor_tasks)
    for task_id in non_anchor_tasks:
        G.add_node(f"Task {task_id}", color='blue')

    # 添加边：非锚点任务指向其对应的锚点（根据 alpha 权重）
    for task_id in non_anchor_tasks:
        alpha = csp_agent.alphas[task_id]
        for idx, weight in enumerate(alpha):
            if weight > 0:
                anchor_task_id = csp_agent.anchor_tasks[idx]
                source_node = f"Anchor {idx}\n(Task {anchor_task_id})"
                target_node = f"Task {task_id}"
                G.add_edge(source_node, target_node, weight=weight)

    # 获取节点颜色
    colors = [G.nodes[node]['color'] for node in G.nodes()]

    # 绘制图形
    pos = nx.spring_layout(G, seed=42)  # 位置布局
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1500, font_size=10)
    # 绘制边的权重
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Task and Anchor Relationship")
    plt.savefig(os.path.join(output_dir, 'task_anchor_relationship.png'))
    plt.close()
