import re

def compute_pnr(seq, gt):
    """
    计算 seq 相对于 gt 的正逆序比 (PNR).
    seq, gt: 长度为 8 的列表, 内容是 0~7 的任意排列.
    返回值: 正逆序比 (float).
    """
    if len(seq) != 8:
        return 0
    
    total_pairs = 0
    match_count = 0
    
    # 对所有下标对 (i, j), i < j
    for i in range(8):
        for j in range(i+1, 8):
            total_pairs += 1
            # 判断输入序列和 GT 序列中 (i, j) 的相对顺序是否一致
            if (seq[i] < seq[j] and gt[i] < gt[j]) or (seq[i] > seq[j] and gt[i] > gt[j]):
                match_count += 1
    
    pnr = match_count / total_pairs  # 共有 28 个 (i, j) 对
    return pnr

def compute_lcs(seq, gt):
    """
    计算 seq 和 gt 的最长公共子序列 (LCS) 长度.
    seq, gt: 长度为 8 的列表, 内容是 0~7 的任意排列.
    返回值: 最长公共子序列长度 (int).
    """
    if len(seq) != 8:
        return 0
    
    # 动态规划表 dp[i][j] 表示 seq[:i] 和 gt[:j] 的 LCS 长度
    # 这里为了便于索引，dp 大小设为 (len(seq)+1) x (len(gt)+1)
    n = len(seq)
    m = len(gt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq[i-1] == gt[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return 1.0/8*dp[n][m]

def compute_position_deviation(seq, gt):
    """
    计算预测序列与真实序列之间的绝对位置偏差。
    seq, gt: 任意长度的列表, 内容是 0~n-1 的任意排列。
    返回值: 偏差得分 (float)，范围 [0, 1]，越小越好。
    """
    if len(seq) != 8:
        return 0
    n = len(seq)
    if len(gt) != n:
        return 0

    # 计算每个元素的偏差
    deviation = 0
    for i in range(n):
        deviation += abs(seq.index(i) - gt.index(i))

    # 将偏差归一化
    max_deviation = n * (n - 1) // 2  # 最大偏差为全错时的累加值
    score = 1 - (deviation / max_deviation)  # 偏差得分，越大表示越接近 gt

    return score

def compute_lcs_with_penalty(seq, gt):
    """
    改进版 LCS，加入匹配位置的惩罚机制。
    seq, gt: 任意长度的列表, 内容是 0~n-1 的任意排列。
    返回值: 改进后的 LCS 奖励值 (float)。
    """
    if len(seq) != 8:
        return 0

    n = len(seq)
    m = len(gt)

    # 动态规划表
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # 构建 LCS DP 表
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[n][m]

    # 惩罚：根据匹配位置的偏差
    penalty = 0
    # for i in range(n):
    #     if seq[i] in gt:
    #         penalty += abs(i - gt.index(seq[i])) / n  # 偏差归一化

    # 归一化 LCS 长度并减去惩罚
    return max(0, (lcs_length / max(n, m)) - penalty)

def compute_pnr_with_penalty(seq, gt):
    """
    改进版 PNR，加入逆序对惩罚机制。
    seq, gt: 任意长度的列表, 内容是 0~n-1 的任意排列。
    返回值: 改进后的正逆序比 (float)。
    """
    n = len(seq)
    assert len(gt) == n

    total_pairs = 0
    match_count = 0
    inversions = 0

    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            # 正逆序匹配
            if (seq[i] < seq[j] and gt[i] < gt[j]) or (seq[i] > seq[j] and gt[i] > gt[j]):
                match_count += 1
            # 计算逆序对
            if seq[i] > seq[j]:
                inversions += 1

    pnr = match_count / total_pairs
    penalty = inversions / total_pairs  # 归一化逆序对数

    # 引入惩罚：奖励 = PNR - Penalty（确保非负）
    return max(0, pnr - penalty)

def extract_rank_think(text: str):
    try:
        # 使用正则：1) 捕获 <think> ... </think> 的内容；2) 匹配 <answer>[...]</answer>
        # 在 [...] 里，匹配任意不含 ] 的字符，非贪婪
        pattern = (
            r'<think>(.*?)</think>\s*'     # 捕获 <think> ... </think> 的内容
            r'<answer>\s*\[([^\]]+)\]\s*</answer>'  # 捕获方括号内的内容
        )
        match = re.search(pattern, text, re.DOTALL)
        if match:
            think_text, list_str = match.groups()
            # 按逗号拆分，并去除空格后转换为 int
            extracted_list = [int(item.strip()) for item in list_str.split(',')]
            return (think_text, extracted_list)
        else:
            return None
    except Exception as e:
        print("Not match:", text, "| Error:", e)
        return None

def rerank_format_reward(predict_str: str) -> float:
    pattern = (
        r'<think>(.*?)</think>\s*'     # 捕获 <think> ... </think> 的内容
        r'<answer>\s*\[([^\]]+)\]\s*</answer>'  # 捕获方括号内的内容
    )
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0

def rerank_count_reward(pred_list: list) -> float:
    if len(pred_list) != 8:
        return 0.0
    
    for i in range(8):
        if i not in pred_list:
            return 0.0

    return 1.0 

def rerank_compute_score(predict_str: str, ground_truth: list):
    """
    综合计算 seq 的奖赏值, 分为两个部分:
    1. 正逆序比 (PNR)
    2. 最长公共子序列 (LCS) 长度
    
    返回 (pnr, lcs_length)
    """
    pred = extract_rank_think(predict_str)
    if pred is not None:
        pred_list = pred[1]
        format_reward = 1
        count_reward = rerank_count_reward(pred_list)

        if count_reward != 0:
            dev_value = compute_position_deviation(pred_list, ground_truth)
            lcs_length = compute_lcs_with_penalty(pred_list, ground_truth)
        else:
            dev_value = 0
            lcs_length = 0
        reward = 0.25 * format_reward + 0.25 * count_reward + 0.5 * lcs_length
    else:
        reward = 0

    return reward

# 测试示例
if __name__ == "__main__":
    input_seq = [7, 6, 5, 4, 3, 2, 1, 0]
    gt_seq = [2, 3, 5, 4, 0, 6, 7, 1]
    
    rewards = reward(input_seq, gt_seq)
    print(rewards)