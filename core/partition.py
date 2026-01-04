import math

def separate_partitions(nodes, capacity):
    if len(nodes) <= capacity:
        return [nodes]

    # Step 1: Determine splitting axis
    max_lat = max(node.lat for node in nodes)
    min_lat = min(node.lat for node in nodes)
    max_lng = max(node.lng for node in nodes)
    min_lng = min(node.lng for node in nodes)

    delta_lat = max_lat - min_lat
    delta_lng = max_lng - min_lng

    if delta_lng >= delta_lat:
        split_axis = 'lng'
    else:
        split_axis = 'lat'

    nodes.sort(key=lambda node: getattr(node, split_axis))

    # Step 2: Identify splitting position following Eq. 7
    # 计算总的容量单位数
    num_units = math.ceil(len(nodes) / capacity)
    # 按照论文 Eq. 7，确保分割位置使得左右子集的节点数都是capacity的整数倍
    split_pos = capacity * (num_units // 2)

    # Step 3: Partition and recurse
    nodes1 = nodes[:split_pos]
    nodes2 = nodes[split_pos:]

    p1 = separate_partitions(nodes1, capacity)
    p2 = separate_partitions(nodes2, capacity)

    return p1 + p2
