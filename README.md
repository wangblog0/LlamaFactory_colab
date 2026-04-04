# LlamaFactory Colab 微调工具包

> 在 Google Colab 上用 [LlamaFactory](https://github.com/hiyouga/LlamaFactory) 对 Qwen3.5 系列模型进行 QLoRA 微调，并与 [WeClone](https://github.com/wangblog0/WeClone) 数据流打通，支持训练后 AWQ 量化与 vLLM 推理部署。

---

## 项目简介

本仓库提供一套完整的、面向 Google Colab 免费 / 付费 GPU 的大语言模型微调流水线，核心功能如下：

| 功能模块 | 说明 |
|---|---|
| **数据准备** | 从 WeClone 生成的 SFT 数据集无缝接入 LlamaFactory |
| **QLoRA 微调** | 适配 T4（16 GB）和更大显存 GPU 的训练配置 |
| **模型导出** | 导出 LoRA 权重并上传到 Hugging Face Hub |
| **AWQ 量化** | 基于 llm-compressor 的 W4A16 量化，输出 compressed-tensors 格式 |
| **vLLM 推理** | 以 OpenAI 兼容 API 格式在 Colab 上启动推理服务，支持 ngrok 公网映射 |

---

## 文件结构

```
LlamaFactory_colab/
├── workflow.ipynb                    # 主流程：数据准备 → SFT 微调 → 模型导出/上传
├── calibration_and_awq_workflow.ipynb # AWQ W4A16 量化流程（llm-compressor）
├── reasoning.ipynb                   # vLLM 推理 + ngrok 公网部署
├── build_calibration_jsonl.py        # 从 SFT 数据集构建量化校准集
├── qwen3.5_35B_base.yaml             # Qwen3.5-35B-A3B-Base QLoRA 训练配置
└── qwen3.5_9B_base_t4.yaml           # Qwen3.5-9B-Base QLoRA 训练配置（T4 优化）
```

---

## 快速开始

### 前提条件

- Google Colab（建议 T4 或以上 GPU；35B 模型需要 A100/L4）
- 已安装 `uv`（Colab 中会自动安装）
- Hugging Face 账号（用于下载模型和上传权重）
- Weights & Biases 账号（可选，用于训练监控）
- ngrok 账号（可选，用于公网推理）

---

### 流程一：WeClone + LlamaFactory SFT 微调

打开 `workflow.ipynb`，按 Cell 顺序执行：

1. **挂载 Google Drive** — 从云盘读取聊天数据
2. **安装 WeClone 依赖** — 克隆并安装 WeClone 项目
3. **生成 SFT 数据集** — 运行 `weclone-cli make-dataset` 生成 `sft-my.json`
4. **安装 LlamaFactory** — 克隆并安装 LlamaFactory（使用独立 uv venv）
5. **下载基础模型** — 通过 `hf download` 拉取 Qwen3.5-9B-Base（或 35B）
6. **启动微调训练** — 调用 `llamafactory-cli train` 并使用预设 YAML 配置
7. **（可选）启动 WebUI** — `llamafactory-cli webui` 进行可视化配置
8. **导出模型权重** — 复制 LoRA adapter 到 Google Drive
9. **上传到 Hugging Face** — `upload_folder` 推送到个人模型仓库

> **多模态数据清洗提示**：如需去除数据中的图片/视频消息，在生成数据集后运行：
> ```bash
> python -m weclone.utils.strip_multimodal_from_sft \
>     --input dataset/res_csv/sft/sft-my.json \
>     --output dataset/res_csv/sft/sft-my-text-only.json
> ```

---

### 流程二：AWQ 量化

打开 `calibration_and_awq_workflow.ipynb`，按步骤执行：

1. **配置路径变量** — 设置合并模型路径、数据集路径、量化输出路径
2. **安装依赖** — `llmcompressor`, `transformers`, `vllm` 等
3. **生成校准集** — 调用 `build_calibration_jsonl.py` 从 SFT 数据采样
4. **执行 AWQ 量化** — W4A16，忽略 `lm_head`，输出 compressed-tensors 格式
5. **启动 vLLM 服务** — 使用 `--quantization compressed-tensors` 加载量化模型
6. **API 测试** — `curl` 验证 OpenAI 兼容接口

常见 OOM 解决思路：

| 问题 | 建议 |
|---|---|
| 量化阶段 OOM | 将 `NUM_CALIBRATION_SAMPLES` 从 256 降到 128 或 64 |
| 量化阶段序列太长 OOM | 将 `MAX_CALIB_SEQ_LEN` 从 512 降到 384 或 256 |
| vLLM 启动 OOM | 将 `--max-model-len` 从 1024 降到 768 或 512 |
| vLLM 量化格式不匹配 | 确认使用 `--quantization compressed-tensors` |

---

### 流程三：vLLM 推理 + 公网部署

打开 `reasoning.ipynb`：

1. 安装 `vllm` 和 `pyngrok`
2. 配置 ngrok AuthToken 开启公网隧道
3. 设置 Hugging Face 模型路径与自定义 API Key
4. 后台启动 vLLM OpenAI API Server

---

## 训练配置说明

### `qwen3.5_9B_base_t4.yaml`（T4 16GB 显存）

| 参数 | 值 | 说明 |
|---|---|---|
| 模型 | `Qwen/Qwen3.5-9B-Base` | 9B 参数基础模型 |
| 微调方式 | LoRA (rank=2, alpha=4) | 显存友好的最小配置 |
| 量化 | QLoRA (NF4, bnb) | 4-bit 加载以节省显存 |
| 批大小 | batch=1, 梯度累积=64 | 等效批大小 64 |
| 精度 | FP16 + Flash Attention 2 | T4 不支持 BF16 |

### `qwen3.5_35B_base.yaml`（A100/L4 大显存）

| 参数 | 值 | 说明 |
|---|---|---|
| 模型 | `Qwen/Qwen3.5-35B-A3B-Base` | 35B MoE 基础模型 |
| 微调方式 | LoRA (rank=8, alpha=16) | q/k/v/o 全量 LoRA |
| 量化 | QLoRA (NF4, bnb) | 4-bit 加载 |
| 批大小 | batch=1, 梯度累积=32 | 等效批大小 32 |
| 精度 | BF16 + Flash Attention 2 | 大显存 GPU 推荐 |

两份配置均使用 `default_system: 请你扮演一名人类，不要说自己是人工智能`，适用于 WeClone 风格的个人 AI 克隆场景。

---

## `build_calibration_jsonl.py` 用法

```bash
python build_calibration_jsonl.py \
    --input  <SFT数据集路径>.json \
    --output calibration.jsonl \
    --num-samples 512 \
    --seed 42 \
    --max-chars 4096 \
    --min-chars 20
```

支持的输入格式：`.json`（列表或含 `data`/`train` 等键的字典）、`.jsonl`。
自动识别 `text`、`prompt`、`instruction`、`conversations`/`messages`（ShareGPT 格式）等字段。

---

## 依赖关系

```
WeClone  ──────────────────────────────────────────────┐
                                                       ▼
                                          SFT 数据集 (sft-my.json)
                                                       │
                                    ┌──────────────────┴────────────────┐
                                    ▼                                   ▼
                          LlamaFactory (SFT 微调)          build_calibration_jsonl.py
                                    │                                   │
                                    ▼                                   ▼
                            LoRA Adapter                      calibration.jsonl
                                    │                                   │
                                    └──────────────────┬────────────────┘
                                                       ▼
                                            llm-compressor AWQ 量化
                                                       │
                                                       ▼
                                              vLLM 推理服务
```

---

## 相关项目

- [LlamaFactory](https://github.com/hiyouga/LlamaFactory) — 统一的 LLM 微调框架
- [WeClone](https://github.com/wangblog0/WeClone) — 基于聊天记录构建个人 AI 克隆
- [llm-compressor](https://github.com/vllm-project/llm-compressor) — vLLM 官方量化工具
- [vLLM](https://github.com/vllm-project/vllm) — 高性能 LLM 推理引擎

---

## License

MIT
