import swanlab
from spectrumlab.models import Qwen_2_5_VL_32B
from spectrumlab.benchmark.signal_group import SignalGroup
from spectrumlab.benchmark.generation_group import GenerationGroup
from spectrumlab.benchmark.perception_group import PerceptionGroup
from spectrumlab.benchmark.semantic_group import SemanticGroup
from spectrumlab.evaluator.choice_evaluator import ChoiceEvaluator
from spectrumlab.evaluator.open_evaluator import OpenEvaluator

# ABLATION_CONFIGS = [
#     {
#         "model_class": Qwen_2_5_VL_72B,
#         "model_name": "Qwen-2.5-VL-72B",
#         "temperature": 0,
#         "top_p": 1.0,
#         "save_dir": "./ablation_qwen_2_5_vl_72b_temp_0_evaluation_results",
#     },
# ]

ABLATION_CONFIGS = [
    {
        "model_class": Qwen_2_5_VL_32B,
        "model_name": "Qwen-2.5-VL-32B",
        "temperature": 1,
        "top_p": 1,
        "save_dir": "./ablation_qwen_2_5_vl_32b_baselines_evaluation_results",
    },
]

# 定义每个 Group 及其子任务和评测器 - 先测试Signal组
GROUPS = [
    {
        "name": "Signal",
        "group": SignalGroup("data"),
        "evaluator": ChoiceEvaluator(),
        "subcategories": None,  # None 表示全部
    },
    {
        "name": "Perception",
        "group": PerceptionGroup("data"),
        "evaluator": ChoiceEvaluator(),
        "subcategories": None,
    },
    {
        "name": "Semantic",
        "group": SemanticGroup("data"),
        "evaluator": ChoiceEvaluator(),
        "subcategories": None,
    },
    {
        "name": "Generation",
        "group": GenerationGroup("data"),
        "evaluator": OpenEvaluator(),
        "subcategories": None,
    },
]

for config in ABLATION_CONFIGS:
    print(f"\n{'='*60}")
    print(
        f"开始消融实验: {config['model_name']} (temperature={config['temperature']}, top_p={config['top_p']})"
    )
    print(f"{'='*60}")

    model = config["model_class"]()

    # 初始化 SwanLab
    swanlab.init(
        workspace="SpectrumLab",
        project="spectrumlab-ablation",
        experiment_name=f"{config['model_name']}_temp_{config['temperature']}_top_p_{config['top_p']}",
        config=config,
    )

    # 遍历每个评测组
    for group_info in GROUPS:
        name = group_info["name"]
        group = group_info["group"]
        evaluator = group_info["evaluator"]
        subcategories = group_info["subcategories"]
        print(f"\n===== Evaluating {name} Group =====")
        data = group.get_data_by_subcategories(subcategories or "all")

        class ModelWithSamplingParams:
            def __init__(self, base_model, temperature, top_p):
                self.base_model = base_model
                self.temperature = temperature
                self.top_p = top_p
                self.model_name = base_model.model_name

            def generate(self, prompt, max_tokens=512):
                return self.base_model.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

        wrapped_model = ModelWithSamplingParams(
            model, config["temperature"], config["top_p"]
        )

        results = evaluator.evaluate(
            data_items=data, model=wrapped_model, save_path=config["save_dir"]
        )

        # Handle both scoring mode (avg_score) and classification mode (accuracy)
        overall_metrics = results["metrics"]["overall"]
        if "avg_score" in overall_metrics:
            # Generation/Scoring mode - use average score
            avg_score = overall_metrics["avg_score"]
            print(
                f"{name} Group evaluation completed! Overall average score: {avg_score:.3f}\n"
            )
            swanlab.log({f"{name}_avg_score": avg_score})
        elif "accuracy" in overall_metrics:
            # Classification mode - use accuracy
            accuracy = overall_metrics["accuracy"]
            print(
                f"{name} Group evaluation completed! Overall accuracy: {accuracy:.2f}%\n"
            )
            swanlab.log({f"{name}_accuracy": accuracy})
        else:
            print(f"{name} Group evaluation completed! No metrics available.\n")

    swanlab.finish()
    print(f"\n消融实验 {config['model_name']} 完成!")
    print(f"结果保存在: {config['save_dir']}")

# use nohup in the terminal to start the evaluation
# nohup python run_ablation_experiments.py > run_ablation.log 2>&1 &
