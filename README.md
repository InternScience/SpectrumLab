<!-- # SpectrumLab -->

<div align="center">
  <img src="docs/public/spectrumlab.svg" alt="SpectrumLab" width="600"/>
  
  <p><strong>A pioneering unified platform designed to systematize and accelerate deep learning research in spectroscopy.</strong></p>
</div>

## 🚀 Quick Start

### Environment Setup

We recommend using conda and uv for environment management:

```bash
# Clone the repository
git clone https://github.com/little1d/SpectrumLab.git
cd SpectrumLab

# Create conda environment
conda create -n spectrumlab python=3.10
conda activate spectrumlab

pip install uv
uv pip install -e .
```

### Data Setup

Download benchmark data from Hugging Face:

- [SpectrumBench v1.0](https://huggingface.co/SpectrumWorld/spectrumbench_v_1.0/tree/main)

Extract the data to the `data` directory in the project root.

### API Keys Configuration

```bash
# Copy and edit environment configuration
cp .env.example .env
# Configure your API keys in the .env file
```

## 💻 Usage

### Python API

```python
from spectrumlab.benchmark import get_benchmark_group
from spectrumlab.models import GPT4o
from spectrumlab.evaluator import get_evaluator

# Load benchmark data
benchmark = get_benchmark_group("perception")
data = benchmark.get_data_by_subcategories("all")

# Initialize model
model = GPT4o()

# Get evaluator
evaluator = get_evaluator("perception")

# Run evaluation
results = evaluator.evaluate(
    data_items=data,
    model=model,
    save_path="./results"
)

print(f"Overall accuracy: {results['metrics']['overall']['accuracy']:.2f}%")
```

### Command Line Interface

The CLI provides a simple way to run evaluations:

```bash
# Basic evaluation
spectrumlab eval --model gpt4o --level perception

# Specify data path and output directory
spectrumlab eval --model claude --level signal --data-path ./data --output ./my_results

# Evaluate specific subcategories
spectrumlab eval --model deepseek --level semantic --subcategories "IR_spectroscopy" "Raman_spectroscopy"

# Customize output length
spectrumlab eval --model internvl --level generation --max-length 1024

# Get help
spectrumlab eval --help
```

## 🤝 Contributing

We welcome community contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Citation

If you use SpectrumLab in your research, please cite our paper:

```bibtex
@inproceedings{10.1145/3770855.3818936,
author = {Yang, Zhuo and Xie, Jiaqing and Shen, Shuaike and Wang, Daolang and Chen, Yeyun and Gao, Ben and Sun, Shuzhou and Qi, Biqing and Zhou, Dongzhan and BAI, LEI and Chen, Linjiang and Zhang, Shufei and Gu, Qinying and Jiang, Jun and Fu, Tianfan and Li, Yuqiang},
title = {SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy},
year = {2026},
isbn = {9798400722592},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3770855.3818936},
doi = {10.1145/3770855.3818936},
abstract = {Deep learning holds immense promise for spectroscopy, yet research and evaluation in this emerging field often lack standardized formulations. To address this issue, we introduce SpectrumWorld, a unified infrastructure for AI-driven spectroscopy. SpectrumWorld consists of SpectrumLab, a pioneering unified platform designed to systematize and accelerate deep learning research in spectroscopy; SpectrumAnnotator, an annotation and curation module that generates high-quality benchmarks from limited seed data; and SpectrumVQA, a multi-layered benchmark suite covering 14 spectroscopic tasks and over 10 spectrum types, featuring spectra curated from over 1.2 million distinct chemical substances. Thorough empirical studies on SpectrumVQA with 23 cutting-edge multimodal LLMs reveal critical limitations of current approaches. We hope SpectrumWorld will serve as a crucial foundation for future advancements in deep learning-driven spectroscopy. Code is released at https://github.com/InternScience/SpectrumLab.},
booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
pages = {12655–12666},
numpages = {12},
keywords = {spectroscopy, deep learning, benchmark, multimodal large language models},
location = {Republic of Korea},
series = {KDD '26}
}
```

## Acknowledgments

- **Experiment Tracking**: [SwanLab](https://github.com/SwanHubX/SwanLab/) for experiment management and visualization
- **Choice Evaluator Framework**: Inspired by [MMAR](https://github.com/ddlBoJack/MMAR)
