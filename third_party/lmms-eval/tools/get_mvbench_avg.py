import argparse
import json
import os
import pdb
import re
from copy import deepcopy
from pathlib import Path

import numpy as np


# read json files
def read_json(path):
    with open(path, "r") as fin:
        datas = json.load(fin)
    return datas


def write_json(path, data):
    with open(path, "w") as fout:
        json.dump(data, fout)
    print("The format file has been saved at:{}".format(path))
    return


def extract_time(paragraph):
    prompt = "A specific example is : 20.8 - 30.0 seconds".lower()
    paragraph = paragraph.lower().replace(prompt, "").replace("to", "-")
    # Split text into sentences based on common delimiters
    sentences = re.split(r"[!?\n]", paragraph)

    # Keywords that might indicate the presence of time information
    keywords = ["starts", "ends", "happens in", "start time", "end time", "start", "end", "happen"]
    # filter sentences by keywords
    candidates = []
    for sentence in sentences:
        # If sentence contains one of the keywords
        if any(keyword in sentence for keyword in keywords):
            candidates.append(sentence)

    timestamps = []
    # Check for The given query happens in m - n (seconds)
    patterns = [r"(\d+\.*\d*)\s*-\s*(\d+\.*\d*)"]

    for time_pattern in patterns:
        time_matches = re.findall(time_pattern, paragraph)
        if time_matches:
            timestamps = [[float(start), float(end)] for start, end in time_matches]

    if len(sentences) == 0:
        return []
    # check for other formats e.g.:
    # 1 .Starting time: 0.8 seconds
    # Ending time: 1.1 seconds
    # 2. The start time for this event is 0 seconds, and the end time is 12 seconds.
    if len(timestamps) == 0:
        times = []
        time_regex = re.compile(r"\b(\d+\.\d+\b|\b\d+)\b")  # time formats (e.g., 18, 18.5)
        for sentence in candidates:
            time = re.findall(time_regex, sentence)
            if time:
                time_in_sec = float(time[0])
                times.append(time_in_sec)
        times = times[: len(times) // 2 * 2]
        timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
    # Check for  examples like:
    # 3. The event 'person flipped the light switch near the door' starts at 00:00:18 and ends at 00:00:23.
    if len(timestamps) == 0:
        times = []
        time_regex = re.compile(r"\b((\d{1,2}:\d{2}:\d{2}))\b")  # time formats (e.g., 18:00, 00:18:05)
        for sentence in candidates:
            time = re.findall(time_regex, sentence)
            if time:
                t = time[0]
            else:
                continue
            # If time is in HH:MM:SS format, convert to seconds
            if t.count(":") == 2:
                h, m, s = map(int, t.split(":"))
                time_in_sec = h * 3600 + m * 60 + s
            elif t.count(":") == 1:
                m, s = map(int, t.split(":"))
                time_in_sec = m * 60 + s
            times.append(time_in_sec)
        times = times[: len(times) // 2 * 2]
        timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
    results = []
    for start, end in timestamps:
        if end > start:
            results.append([start, end])
        else:
            results.append([end, start])
    if len(results) > 1:
        results = results[:1]
    return results


def iou(A, B):
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    return max(min1 - max0, 0) / (max1 - min0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", default="your_result.json")
    args = parser.parse_args()

    datas = read_json(args.f)

    score = [v.get('mvbench_accuracy,none', 0) for k, v in datas['results'].items()]
    score = sum(score) / 20

    print(f"Average Score: {score}")
