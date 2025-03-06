import re
import json

def temporal_iou(A, B):
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    _iou=max(min1 - max0, 0) / (max1 - min0)
    return max(0,_iou)

def tvg_format_reward(predict_str: str) -> float:
    pattern = r'<think>.*?</think>\s*<answer>\s*{\s*"start_time"\s*:\s*"([\d.]+)"\s*,\s*"end_time"\s*:\s*"([\d.]+)"\s*}\s*</answer>'
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0


def tvg_accuracy_reward(predict_str: str, ground_truth: list, video_length: float) -> float:
    try:
        content_answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        content_answer = content_answer_match.group(1).strip()
        answer_data = json.loads(content_answer)

        answer_timestamp = [
            float(answer_data["start_time"]) / video_length, 
            float(answer_data["end_time"]) / video_length
        ]
        reward = temporal_iou(answer_timestamp, ground_truth)

        # print(f"[pred]: {answer_timestamp} [gt]: {ground_truth} [reward]: {reward}")
        # print(f"gt: {ground_truth}, pred: {answer_timestamp}")

        # reward = 1.0 if reward >= 0.5 else 0
        return reward
    except Exception as e:
        # print(e)
        pass  # Continue to next verification method if this fails

    return 0.0


def tvg_compute_score(predict_str: str, ground_truth: list, video_length: float) -> float:
    # print(predict_str, ground_truth, video_length)
    acc_reward = tvg_accuracy_reward(predict_str, ground_truth, video_length)
    format_reward = tvg_format_reward(predict_str)
    # print(f"acc: {acc_reward}, format: {format_reward}")
    reward = 0.9 * acc_reward + 0.1 * format_reward
    # reward /= 2
    return reward
