
from core.partition import separate_partitions

# 定义一个简单的 Node 类，模拟传感器
class Node:
    def __init__(self, id, lat, lng):
        self.id = id
        self.lat = lat
        self.lng = lng

    def __repr__(self):
        return f"Node(id={self.id}, lat={self.lat}, lng={self.lng})"


def run_test():
    # 1. 造数据：模拟 6 个传感器节点
    # 我们故意造一些分布，看看它会不会切
    nodes = [
        Node(0, lat=10, lng=10),
        Node(1, lat=11, lng=10),  # 和 0 很近
        Node(2, lat=20, lng=20),
        Node(3, lat=21, lng=20),  # 和 2 很近
        Node(4, lat=30, lng=30),
        Node(5, lat=31, lng=30)  # 和 4 很近
    ]

    # 2. 设定容量 C
    capacity = 2  # 每个 Patch 最多 2 个节点

    print(f"原始节点数量: {len(nodes)}")
    print(f"设定容量 C: {capacity}")

    # 3. 运行你的算法
    patches = separate_partitions(nodes, capacity)

    # 4. 检查结果
    print("\n--------- 划分结果 ---------")
    print(f"生成的 Patch 总数: {len(patches)}")

    for i, patch in enumerate(patches):
        print(f"Patch {i}: 包含 {len(patch)} 个节点 -> {[n.id for n in patch]}")

        # 验证 1: 容量限制
        if len(patch) > capacity:
            print("❌ 错误：这个 Patch 超过了容量限制！")
            return

    # 验证 2: 节点总数
    total_nodes = sum(len(p) for p in patches)
    if total_nodes != len(nodes):
        print(f"❌ 错误：节点总数不对！原来有 {len(nodes)} 个，现在只有 {total_nodes} 个。")
    else:
        print("\n✅ 测试通过！节点数量正确，容量限制正确。")


if __name__ == "__main__":
    run_test()