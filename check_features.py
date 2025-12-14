import numpy as np
import os


def check_feature_semantics(data_dir):
    his_path = os.path.join(data_dir, 'his.npz')

    try:
        raw = np.load(his_path, allow_pickle=True)
        data = raw['data']  # (T, N, 3) Normalized
        mean = raw['mean']  # Shape likely (1, 1, 3) or (3,)
        std = raw['std']  # Shape likely (1, 1, 3) or (3,)

        print(f"数据加载成功: {his_path}")
        print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")

        # 1. 反归一化 (Recover Physical Values)
        # Formula: Real = Norm * Std + Mean
        # 我们只取前 1000 个时间步来分析，节省内存
        sample_data = data[:1000]
        real_data = sample_data * std + mean

        # 2. 逐个通道分析统计特性
        feature_names = ["Channel 0", "Channel 1", "Channel 2"]

        print("\n====== 特征物理意义解密 ======")
        for i in range(3):
            # 获取第 i 个特征的所有数据
            feat_vals = real_data[..., i].flatten()

            # 过滤掉 0 值 (0通常是空缺或深夜无车，影响均值判断)
            non_zero_vals = feat_vals[feat_vals > 0.1]

            f_mean = np.mean(non_zero_vals) if len(non_zero_vals) > 0 else 0
            f_max = np.max(feat_vals)
            f_min = np.min(feat_vals)

            print(f"[{feature_names[i]}]")
            print(f"  -> 均值 (Mean): {f_mean:.2f}")
            print(f"  -> 最大值 (Max):  {f_max:.2f}")
            print(f"  -> 最小值 (Min):  {f_min:.2f}")

            # 3. 智能推断
            guess = "未知"
            # 流量 (Flow): 通常均值几百，最大值可能过千 (5分钟通过的车辆数)
            if f_max > 100 and f_mean > 20:
                guess = "流量 (Flow) - 车辆数/5min"
            # 速度 (Speed): 高速公路通常均值在 60-70 英里/小时，最大值不超过 100
            elif 40 < f_mean < 80 and f_max < 120:
                guess = "速度 (Speed) - mph"
            # 占有率 (Occupancy): 通常是 0-1 之间的小数，或者 0-100 的百分比
            # PeMS有时输出 0.15 这种，有时输出 15.0
            elif f_max <= 1.0 or (f_mean < 10 and f_max <= 100):
                guess = "占有率 (Occupancy) - % 或 比例"

            print(f"  => 推测身份: 【 {guess} 】\n")

    except Exception as e:
        print(f"出错: {e}")


if __name__ == "__main__":
    # 修改为你的路径
    target_dir = './data/sd/2019'
    check_feature_semantics(target_dir)