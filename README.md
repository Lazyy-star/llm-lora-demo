## GPT-2 LoRA 微调示例项目

这是一个使用 Hugging Face Transformers + PEFT 实现的 GPT-2 小模型微调项目。通过 LoRA（低秩适配器）在少量自定义文本上微调模型，实现特定文本生成任务，适合学习和展示 LLM 微调流程。

### 项目结构
llm-lora-demo/
├── train.py # 微调训练脚本
├── inference.py # 测试生成脚本
├── data/ # 放训练样例数据
├── lora_model/ # 保存微调后的 LoRA 权重
├── README.md
└── .gitignore

### 环境依赖
Python 3.10+
torch
transformers
datasets
peft

推荐使用虚拟环境安装依赖：
python -m venv venv
venv\Scripts\activate # Windows
pip install torch transformers datasets peft

### 使用方法
1. 准备数据
在 data/ 文件夹下放训练文本文件（例如 train.json 或 txt 文件），每条文本一行。
2. 训练微调模型
在终端运行：python train.py
训练完成后，LoRA 权重会保存到 lora_model/ 文件夹。
3. 生成文本
在终端运行：
python inference.py
使用微调后的 LoRA 权重生成文本，验证微调效果。

