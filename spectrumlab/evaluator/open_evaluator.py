import re
from typing import Dict, Any, Optional
from .base import BaseEvaluator
from spectrumlab.utils.image_utils import (
    prepare_images_for_prompt,
    normalize_image_paths,
)
from spectrumlab.models import GPT4o
from tqdm import tqdm


class OpenEvaluator(BaseEvaluator):
    def __init__(
        self,
        prediction_key: str = "model_prediction",
        score_model: Optional[Any] = None,
    ):
        super().__init__(prediction_key)
        # 支持自定义评分模型，默认 GPT4o
        self.score_model = score_model or GPT4o()

    def _build_prompt(self, item: Dict) -> Any:
        """
        为被测模型构建解题 prompt。
        """
        question = item.get("question", "")
        images = normalize_image_paths(item.get("image_path"))
        text_content = f"Question: {question}\nPlease answer the question."
        if images:
            assert all(
                isinstance(p, str) for p in images
            ), f"images should be List[str], got {images}"
            return {"text": text_content, "images": prepare_images_for_prompt(images)}
        else:
            return text_content

    def _build_score_prompt(self, item: Dict, model_output: Any) -> Any:
        """
        构建评分 prompt，包含评分准则。
        """
        question = item.get("question", "")
        images = normalize_image_paths(item.get("image_path"))
        reference_answer = item.get("answer", "")
        # 支持图片型参考答案
        reference_images = []
        if isinstance(reference_answer, str) and reference_answer.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
        ):
            reference_images = [reference_answer]
            reference_answer_text = "[See reference image]"
        else:
            reference_answer_text = reference_answer
        # 支持图片型模型输出
        model_output_images = []
        if isinstance(model_output, str) and model_output.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
        ):
            model_output_images = [model_output]
            model_output_text = "[See model output image]"
        else:
            model_output_text = model_output

        # ========== 评分准则 v1 (已废弃，保留作为参考) ==========
        # prompt_lines_v1 = [
        #     "You are an expert evaluator. Given the following question, reference answer, and model answer, please rate the model answer on a scale of 0 to 1, and explain your reasoning.",
        #     "Scoring rules:",
        #     "- If the reference answer is an image but the model output does not contain an image, score 0.",
        #     "- If the reference answer is text but the model output does not contain text, score 0.",
        #     "- Otherwise, score based on the similarity and correctness of the model output compared to the reference answer.",
        #     "- If both text and image are present, consider both in your evaluation.",
        #     "Please output your score in the format: \\score{X}, where X is a number between 0 and 1.",
        #     "",
        #     f"Question: {question}",
        # ]

        # ========== 评分准则 v2 (优化版本) ==========
        prompt_lines = [
            "You are an expert evaluator. Given the following question, reference answer, and model answer. Rate the model answer on a scale of 0-1 and provide a BRIEF explanation.",
            "",
            "Scoring Rules:",
            "",
            "1. Modality:",
            "   - Reference answers are images. Model outputs may be images or text descriptions.",
            "   - If model outputs image (matching modality): max score 0.8",
            "   - If model outputs text (different modality): text descriptions are VALID and should be evaluated, max score 0.8",
            "   - Note: Many models cannot output images; text descriptions are acceptable alternatives.",
            "",
            "2. Scale (0.1 increments):",
            "   - 0.0: Incorrect/irrelevant/fails to address question",
            "   - 0.1-0.2: Mostly incorrect, minimal relevance",
            "   - 0.3-0.4: Partially correct, significant errors/omissions",
            "   - 0.5-0.6: Moderately correct, missing key information",
            "   - 0.7-0.8: Mostly correct, minor errors",
            "   - 0.9-1.0: Excellent to perfect (1.0 reserved for flawless answers with matching modality)",
            "",
            "3. Evaluation Criteria (weight: Correctness 60%, Completeness 25%, Relevance 15%):",
            "   - CRITICAL: Focus on FINAL ANSWER accuracy, not reasoning process",
            "   - CONCISENESS: Long explanations do NOT earn higher scores",
            "   - STRICTNESS: High scores (0.7+) are ONLY for genuinely accurate and complete answers",
            "",
            "4. Guidelines:",
            "   - Be STRICT: High scores (≥0.7) only for genuinely accurate answers",
            "   - Wrong final answer = low score, regardless of reasoning quality",
            "   - Use discrete scores (0.0, 0.1, 0.2, ..., 1.0)",
            "",
            "Output format: \\score{X} where X ∈ [0.0, 1.0]",
            "",
            f"Question: {question}",
        ]
        if images:
            prompt_lines.append("[See question image(s)]")
        prompt_lines.append("")
        prompt_lines.append(f"Reference Answer: {reference_answer_text}")
        if reference_images:
            prompt_lines.append("[See reference answer image(s)]")
        prompt_lines.append("")
        prompt_lines.append(f"Model Output: {model_output_text}")
        if model_output_images:
            prompt_lines.append("[See model output image(s)]")
        prompt_lines.append("")
        prompt_lines.append("Your response:")
        text_content = "\n".join(prompt_lines)
        # 构建多模态输入
        all_images = []
        if images:
            assert all(
                isinstance(p, str) for p in images
            ), f"images should be List[str], got {images}"
            all_images += prepare_images_for_prompt(images)
        if reference_images:
            assert all(
                isinstance(p, str) for p in reference_images
            ), f"reference_images should be List[str], got {reference_images}"
            all_images += prepare_images_for_prompt(reference_images)
        if model_output_images:
            assert all(
                isinstance(p, str) for p in model_output_images
            ), f"model_output_images should be List[str], got {model_output_images}"
            all_images += prepare_images_for_prompt(model_output_images)
        if all_images:
            return {"text": text_content, "images": all_images}
        else:
            return text_content

    def _extract_prediction(self, response: str, item: Dict) -> float:
        """
        提取 \\score{X}
        """
        if not response:
            return 0.0
        score_pattern = r"\\score\{([0-9.]+)\}"
        matches = re.findall(score_pattern, response)
        if matches:
            try:
                score = float(matches[-1])
                return max(0.0, min(1.0, score))
            except Exception:
                return 0.0
        return 0.0

    def _calculate_accuracy(self, answer: Any, prediction: float, item: Dict) -> bool:
        """
        For OpenEvaluator, we use scoring mode instead of binary classification.
        This method returns True for framework compatibility only.
        The actual score (0.0-1.0) is stored directly in the 'pass' field.
        """
        return True

    def evaluate(
        self,
        data_items,
        model,
        max_out_len=512,
        batch_size=None,
        save_path="./eval_results",
        score_model=None,
    ):
        """
        两阶段评测：先用被测模型生成答案，再用评分模型打分。
        支持 score_model 参数。
        """
        score_model = score_model or self.score_model
        results = []
        print("🚀 Running model inference...")
        model_outputs = []
        # 1. 让被测模型生成答案（带进度条）
        for item in tqdm(data_items, desc="Generating responses", unit="item"):
            prompt = self._build_prompt(item)
            model_output = model.generate(prompt, max_out_len)
            model_outputs.append(model_output)
        # 2. 评分阶段（带进度条）
        print("📝 Running scoring model...")
        for item, model_output in tqdm(
            zip(data_items, model_outputs),
            total=len(data_items),
            desc="Scoring responses",
            unit="item",
        ):
            score_prompt = self._build_score_prompt(item, model_output)
            score_response = score_model.generate(score_prompt, max_out_len)
            score = self._extract_prediction(score_response, item)
            # 3. 保存所有信息
            item_result = item.copy()
            item_result[self.prediction_key] = score
            item_result["model_output"] = model_output
            item_result["score_response"] = score_response
            # Store score (0.0-1.0) directly in 'pass' field for avg_score calculation
            item_result["pass"] = score
            results.append(item_result)
        # 4. 保存和统计
        saved_files = self._save_results(results, save_path)
        metrics = self._calculate_metrics(results)
        self._print_results(metrics)
        return {
            "metrics": metrics,
            "saved_files": saved_files,
            "total_items": len(results),
        }
