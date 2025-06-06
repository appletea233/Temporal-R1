# Reinforcement Learning Tuning for VideoLLMs: Reward Design and Data Efficiency

<div align="center">
<img src="./pics/fig1.png" width="1000"/>
</div>

## Project Introduction

Understanding real-world videos with complex semantics and long temporal dependencies remains a fundamental challenge in computer vision. Recent progress in multimodal large language models (MLLMs) has demonstrated strong capabilities in vision-language tasks, while reinforcement learning tuning (RLT) has further improved their reasoning abilities. In this work, we explore RLT as a post-training strategy to enhance the video-specific reasoning capabilities of MLLMs. Built upon the Group Relative Policy Optimization (GRPO) framework, we propose a dual-reward formulation that supervises both semantic and temporal reasoning through discrete and continuous reward signals. To facilitate effective preference-based optimization, we introduce a variance-aware data selection strategy based on repeated inference to identify samples that provide informative learning signals. We evaluate our approach across eight representative video understanding tasks, including VideoQA, Temporal Video Grounding, and Grounded VideoQA. Our method consistently outperforms supervised fine-tuning and existing RLT baselines, achieving superior performance with significantly less training data. These results underscore the importance of reward design and data selection in advancing reasoning-centric video understanding with MLLMs.

## News
* [2025/3/21] 🔥 The initial version of the code has been released! Please check our Hugging Face repo. [[Checkpoints](https://huggingface.co/appletea2333)]  
* [2025/6/2] 📄 The paper has been released! [[Paper](https://arxiv.org/pdf/2506.01908)]  
* [ ] ⏰ New version code, including optimized reward mechanisms and additional datasets, is coming soon.

## Installation Guide
```
git clone https://github.com/appletea233/Temporal-R1.git
cd Temporal-R1
pip install -e .

# eval with lmms-eval
cd third_party/lmms-eval
pip install -e .
```

## Dataset
1. Download the [annotation files and videos](https://huggingface.co/datasets/appletea2333/temporal_r1)

2. You need to create a file named `tvg.yaml` under  `examples/data_config` with the following content:

```
datasets:
    - json_path: xxx.json
      data_folder: xx
    - json_path: yyy.json
      data_folder: yy
```
The json_path is the dataset file, and the data_folder stores the videos.

## Usage Instructions

Train the Model:
```
bash examples/qwen2_5_vl_3b_tvg.sh
```
Run Inference:
```
# Custom Inference
bash third_party/lmms-eval/examples/eval_tvg.sh $GPUS $MODEL_PATH $TASKS

# R1 Inference
bash third_party/lmms-eval/examples/eval_tvg_r1.sh $GPUS $MODEL_PATH $TASKS

# task uses temporal_grounding_charades,temporal_grounding_activitynet
```
## Experimental Results
| Para. | token         | RL  |think | mIoU(Charades)  | mIoU(ANet-tvg, OOD*)       | Checkpoint|
|------|------------------|----------|------------|------------|-----------|-----------|
| 3b    | 2048         | ❌     | ❌ | 37.22     | 18.92 | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)|
| 3b    | 2048         | SFT    | ❌ | 45.95     | 20.86 |[SFT-3B-Charades](https://huggingface.co/appletea2333/SFT-3B-Charades) |
| 3b    | 2048         | ✅    | ❌ | 51.10     | 22.10 |[Temporal-R1-3B-Charades](https://huggingface.co/appletea2333/Temporal-R1-3B-Charades) |
| 3b    | 2048         | ✅     | ✅ | 53.93 <span style="color: green;">(**+7.98**)</span>    | 23.07 <span style="color: green;">(**+2.21**)</span>| [Temporal-R1-3B-Charades](https://huggingface.co/appletea2333/Temporal-R1-3B-Charades) |

***OOD**: Our model is trained exclusively on Charades-tvg, while ANet-tvg represents out-of-domain data.

### 1. Video Temporal Grounding Results

Experimental results demonstrate that, compared to the SFT model, the GRPO-trained model not only achieves significant performance improvements but also exhibits reasoning ("think") capabilities and stronger generalization. Specifically, the mIoU on the Charades dataset increased by **+7.98**, while the mIoU on the ActivityNet benchmark also showed a improvement (**+2.21**). These findings indicate that GRPO training enables the model to perform better in complex tasks and adapt more effectively to diverse data distributions. In addition, we also evaluated the model's performance when generating only the final output without including its reasoning process. The experimental results indicate that performance declines across the board, suggesting that the inclusion of a reasoning process has a positive effect on our model. We plan to release more related experimental results in the future.

### 2.Training Phenomena

<div align="center">
<img src="./pics/reward curve.png" width="400"/> <img src="./pics/token len.png" width="400"/>
</div>
From the left figure, it can be observed that the average reward increases progressively during training and eventually converges to a stable value. This indicates that the reward we design is reasonable and effectively guides the model in optimizing the objective and is improving performance. The right figure illustrates the variation in the token length of responses. Initially, the length increases rapidly, followed by a sharp decline, and then fluctuates upward within a certain range. This phenomenon is consistent with the training characteristics of DeepSeek-Zero, reflecting the model’s adaptive adjustment of length during generation. Such dynamic changes may represent the model's natural behavior in balancing output quality and complexity, further validating the effectiveness and rationality of the training strategy.

### 3. VideoQA Results

We also explored the performance of our model when directly tested on the VideoQA task using **MVBench**. Our model achieves an accuracy of 59.6, slightly lower than the base model's 63.35. However, the model fine-tuned through direct supervised fine-tuning on the same training data completely **lost its ability to output valid options**. This phenomenon highlights that reinforcement learning-based fine-tuning preserves a significantly higher degree of generalization compared to SFT.

## Acknowledgments
We want to thank [EasyR1](https://github.com/hiyouga/EasyR1), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [llama-factory](https://github.com/hiyouga/LLaMA-Factory) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for publicly releasing their code and pretrained models.

## Citation
Contributors: [Hongyu Li](https://scholar.google.com/citations?hl=en&user=PccL82sAAAAJ), [Songhao Han](https://scholar.google.com/citations?hl=en&user=s-exUYYAAAAJ), [Yue Liao](https://scholar.google.com/citations?hl=en&user=mIt-3fEAAAAJ&view_op=list_works&sortby=pubdate), Junfeng Luo, [Jialin Gao](https://scholar.google.com/citations?user=sj4FqEgAAAAJ&hl=zh-CN), [Shuicheng Yan](https://scholar.google.com/citations?user=DNuiPHwAAAAJ&hl=en&oi=ao), [Si Liu](https://scholar.google.com/citations?user=-QtVtNEAAAAJ&hl=zh-CN)
```bibtex
@article{li2025reinforcement,
  title={Reinforcement Learning Tuning for VideoLLMs: Reward Design and Data Efficiency},
  author={Li, Hongyu and Han, Songhao and Liao, Yue and Luo, Junfeng and Gao, Jialin and Yan, Shuicheng and Liu, Si},
  journal={arXiv preprint arXiv:2506.01908},
  year={2025}
}
```
