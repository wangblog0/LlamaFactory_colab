# About

**LlamaFactory Colab** 是一套面向 Google Colab 的 Qwen3.5 系列大语言模型微调工具包，深度集成 [WeClone](https://github.com/wangblog0/WeClone) 数据流，支持从聊天记录到个人 AI 克隆的完整流水线：SFT 微调 → AWQ 量化 → vLLM 推理部署，全程在 Colab 免费 / 付费 GPU 上运行。

## 核心特性

- 🚀 **开箱即用的 Colab Notebook**：三个场景对应三个 Notebook，按 Cell 顺序执行即可
- 🤖 **WeClone 深度集成**：自动读取 WeClone 生成的 SFT 数据并送入 LlamaFactory
- 🪶 **显存友好**：T4（16 GB）可跑 Qwen3.5-9B QLoRA，A100/L4 可跑 35B MoE QLoRA
- 📦 **AWQ 量化**：使用 llm-compressor 完成 W4A16 量化，输出可被 vLLM 直接加载的 compressed-tensors 格式
- 🌐 **公网推理**：vLLM + ngrok，一键将 Colab 实例暴露为 OpenAI 兼容 API
- 📤 **一键上传**：训练完成后直接将 LoRA adapter 推送到 Hugging Face Hub

## 适用场景

- 用自己的聊天记录（微信、QQ 等）训练专属个人 AI 助手
- 在 Colab 免费 GPU 上低成本验证 Qwen3.5 系列模型的 SFT 效果
- 量化导出模型供本地或云端 vLLM 服务使用
