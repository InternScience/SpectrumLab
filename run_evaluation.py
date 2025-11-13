import swanlab
from spectrumlab.models import Intern_S1
from spectrumlab.benchmark.generation_group import GenerationGroup
from spectrumlab.evaluator.open_evaluator import OpenEvaluator

# export your swanlab api-key

# Change your model!
# MODEL = GPT4o()
# For Intern_S1, you can control thinking_mode during initialization:
# MODEL = Intern_S1(thinking_mode=True)   # Enable thinking mode (default)
# MODEL = Intern_S1(thinking_mode=False) # Disable thinking mode
MODEL = Intern_S1(thinking_mode=False)


# Change this!!! such as gpt-4o_evaluation_results
SAVE_DIR = "./intern_s1_generation_no_thinking_mode_evaluation_results"

# 定义每个 Group 及其子任务和评测器
GROUPS = [
    # {
    #     "name": "Signal",
    #     "group": SignalGroup("data"),
    #     "evaluator": ChoiceEvaluator(),
    #     "subcategories": None,  # None 表示全部
    # },
    # {
    #     "name": "Perception",
    #     "group": PerceptionGroup("data"),
    #     "evaluator": ChoiceEvaluator(),
    #     "subcategories": None,
    # },
    # {
    #     "name": "Semantic",
    #     "group": SemanticGroup("data"),
    #     "evaluator": ChoiceEvaluator(),
    #     "subcategories": None,
    # },
    {
        "name": "Generation",
        "group": GenerationGroup("data"),
        "evaluator": OpenEvaluator(),
        "subcategories": None,
    },
]

# Change the experiment_name to your model name!!!
swanlab.init(
    workspace="SpectrumLab",
    project="spectrumlab-v2-generation-eval",
    experiment_name="intern_s1_generation_no_thinking_mode_evaluation_results",
    config={"model": MODEL.model_name},
)

for group_info in GROUPS:
    name = group_info["name"]
    group = group_info["group"]
    evaluator = group_info["evaluator"]
    subcategories = group_info["subcategories"]
    print(f"\n===== Evaluating {name} Group =====")
    data = group.get_data_by_subcategories(subcategories or "all")
    results = evaluator.evaluate(data_items=data, model=MODEL, save_path=SAVE_DIR)

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
        print(f"{name} Group evaluation completed! Overall accuracy: {accuracy:.2f}%\n")
        swanlab.log({f"{name}_accuracy": accuracy})
    else:
        print(f"{name} Group evaluation completed! No metrics available.\n")

swanlab.finish()

# use nohup in the terminal to start the evaluation
# nohup python run_evaluation.py > run_intern_s1_generation_no_thinking_mode.log 2>&1 &
