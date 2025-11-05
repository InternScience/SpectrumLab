from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm


class BaseEvaluator(ABC):
    def __init__(self, prediction_key: str = "model_prediction"):
        self.prediction_key = prediction_key

    @abstractmethod
    def _build_prompt(self, item: Dict) -> str:
        pass

    @abstractmethod
    def _extract_prediction(self, response: str, item: Dict) -> str:
        pass

    @abstractmethod
    def _calculate_accuracy(self, answer: str, prediction: str, item: Dict) -> bool:
        pass

    def evaluate(
        self,
        data_items: List[Dict],
        model,
        max_out_len: int = 512,
        batch_size: Optional[int] = None,
        save_path: str = "./eval_results",
    ) -> Dict:
        if not data_items:
            print("❌ No data items provided")
            return {"error": "No data items provided"}

        print(f"🔄 Starting evaluation on {len(data_items)} items...")
        print(f"📝 Model: {type(model).__name__}")

        # 1. Build prompts
        print("📝 Building prompts...")
        prompts = [self._build_prompt(item) for item in data_items]

        # 2. Run model inference
        print("🚀 Running model inference...")
        responses = []
        try:
            # 统一使用sequential generation with progress bar
            for i, prompt in enumerate(
                tqdm(prompts, desc="Generating responses", unit="item")
            ):
                try:
                    response = model.generate(prompt, max_out_len)
                    responses.append(response)
                except Exception as e:
                    print(f"\n⚠️  Error on item {i + 1}: {e}")
                    responses.append(f"Error: {str(e)}")

        except Exception as e:
            return {"error": f"Model generation failed: {e}"}

        # 3. Extract predictions and add to data
        print("🔍 Extracting predictions...")
        processed_items = []
        for item, response in tqdm(
            zip(data_items, responses),
            desc="Processing responses",
            total=len(data_items),
            unit="item",
        ):
            item_copy = item.copy()
            prediction = self._extract_prediction(response, item)
            item_copy[self.prediction_key] = prediction
            item_copy["model_response"] = response

            answer = item.get("answer", "")
            is_correct = self._calculate_accuracy(answer, prediction, item)
            item_copy["pass"] = is_correct

            processed_items.append(item_copy)

        # 4. Save results
        saved_files = self._save_results(processed_items, save_path)
        print(f"💾 Results saved to: {saved_files}")

        # 5. Calculate metrics
        print("📊 Calculating metrics...")
        metrics = self._calculate_metrics(processed_items)

        # 6. Print results
        self._print_results(metrics)

        return {
            "metrics": metrics,
            "saved_files": saved_files,
            "total_items": len(data_items),
        }

    def evaluate_many(
        self,
        data_items: List[Dict],
        model,
        max_out_len: int = 512,
        batch_size: Optional[int] = None,
        save_path: str = "./eval_results",
        n_jobs: int = -1,
    ) -> Dict:
        """
        Evaluate a single model on data_items with parallel processing.

        Args:
            data_items: List of data items to evaluate
            model: Model instance to evaluate
            max_out_len: Maximum output length for model generation
            batch_size: Batch size for processing (if None, will be auto-calculated)
            save_path: Base path to save results
            n_jobs: Number of parallel jobs (-1 for all available cores)

        Returns:
            Dictionary containing evaluation results
        """
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import math

        if not data_items:
            print("❌ No data items provided")
            return {"error": "No data items provided"}

        # Set number of jobs
        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        # Calculate batch size if not provided
        if batch_size is None:
            batch_size = max(1, math.ceil(len(data_items) / n_jobs))

        print(f"🔄 Starting parallel evaluation on {len(data_items)} items...")
        print(f"📝 Model: {type(model).__name__}")
        print(f"⚡ Using {n_jobs} parallel workers with batch size {batch_size}")

        # Split data into batches
        batches = [
            data_items[i : i + batch_size]
            for i in range(0, len(data_items), batch_size)
        ]

        print(f"📦 Split into {len(batches)} batches")

        # Build prompts for all items
        print("📝 Building prompts...")
        all_prompts = [self._build_prompt(item) for item in data_items]

        # Split prompts into batches
        prompt_batches = [
            all_prompts[i : i + batch_size]
            for i in range(0, len(all_prompts), batch_size)
        ]

        def process_batch(batch_data):
            """Process a batch of prompts and return responses."""
            batch_prompts, batch_indices = batch_data
            batch_responses = []

            for i, prompt in enumerate(batch_prompts):
                try:
                    response = model.generate(prompt, max_out_len)
                    batch_responses.append(response)
                except Exception as e:
                    # 保持与evaluate方法一致的错误处理
                    original_index = batch_indices[i]
                    print(f"\n⚠️  Error on item {original_index + 1}: {e}")
                    batch_responses.append(f"Error: {str(e)}")

            return batch_indices, batch_responses

        # Prepare batch data with indices
        batch_data_list = []
        for i, prompt_batch in enumerate(prompt_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(data_items))
            batch_indices = list(range(start_idx, end_idx))
            batch_data_list.append((prompt_batch, batch_indices))

        # Execute parallel processing
        all_responses = [None] * len(data_items)

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all batch tasks
            future_to_batch = {
                executor.submit(process_batch, batch_data): batch_data[1][0]
                for batch_data in batch_data_list
            }

            # Collect results as they complete
            for future in tqdm(
                as_completed(future_to_batch),
                total=len(future_to_batch),
                desc="Processing batches",
                unit="batch",
            ):
                try:
                    batch_indices, batch_responses = future.result()
                    for idx, response in zip(batch_indices, batch_responses):
                        all_responses[idx] = response
                except Exception as e:
                    print(f"❌ Error processing batch: {e}")

        # Process responses and calculate results
        print("🔍 Processing responses...")
        processed_items = []
        for item, response in tqdm(
            zip(data_items, all_responses),
            desc="Processing responses",
            total=len(data_items),
            unit="item",
        ):
            item_copy = item.copy()
            prediction = self._extract_prediction(response, item)
            item_copy[self.prediction_key] = prediction
            item_copy["model_response"] = response

            answer = item.get("answer", "")
            is_correct = self._calculate_accuracy(answer, prediction, item)
            item_copy["pass"] = is_correct

            processed_items.append(item_copy)

        # Save results
        saved_files = self._save_results(processed_items, save_path)
        print(f"💾 Results saved to: {saved_files}")

        # Calculate metrics
        print("📊 Calculating metrics...")
        metrics = self._calculate_metrics(processed_items)

        # Print results
        self._print_results(metrics)

        return {
            "metrics": metrics,
            "saved_files": saved_files,
            "total_items": len(data_items),
            "parallel_info": {
                "n_jobs": n_jobs,
                "batch_size": batch_size,
                "n_batches": len(batches),
            },
        }

    def _save_results(self, results_data: List[Dict], save_path: str) -> List[str]:
        """Save results grouped by subcategory. If save_path is None, do not save."""
        if not results_data or save_path is None:
            return []

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Group by subcategory
        subcategory_data = {}
        for item in results_data:
            sub_category = item.get("sub_category", "Unknown")
            if sub_category not in subcategory_data:
                subcategory_data[sub_category] = []
            subcategory_data[sub_category].append(item)

        # Save each subcategory
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for sub_category, data_list in subcategory_data.items():
            safe_name = sub_category.replace(" ", "_").replace("/", "_")
            filename = f"{safe_name}_{timestamp}.json"
            filepath = save_dir / filename

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data_list, f, indent=2, ensure_ascii=False)
                saved_files.append(str(filepath))
                print(f"  ✅ Saved {len(data_list)} items to {filename}")
            except Exception as e:
                print(f"❌ Failed to save {sub_category}: {e}")

        return saved_files

    def _calculate_metrics(self, processed_items: List[Dict]) -> Dict:
        if not processed_items:
            return {}

        # Overall metrics
        total = len(processed_items)
        no_prediction = 0
        total_score = 0.0

        # Detect evaluation mode based on first item's pass field type
        # - Scoring mode (OpenEvaluator): pass is float (0.0-1.0)
        # - Classification mode (ChoiceEvaluator): pass is bool (True/False)
        first_pass_value = processed_items[0].get("pass", False)
        use_scoring_mode = isinstance(
            first_pass_value, (int, float)
        ) and not isinstance(first_pass_value, bool)

        # Only used in classification mode
        correct = 0

        # Category and subcategory metrics
        category_stats = {}
        subcategory_stats = {}

        for item in processed_items:
            prediction = item.get(self.prediction_key, "")
            category = item.get("category", "Unknown")
            sub_category = item.get("sub_category", "Unknown")

            # Check if prediction exists
            if prediction is None or (
                isinstance(prediction, str) and prediction.strip() == ""
            ):
                no_prediction += 1

            pass_value = item.get("pass", False)

            if use_scoring_mode:
                # Scoring mode (OpenEvaluator): use score directly
                score = float(pass_value)
                total_score += score
            else:
                # Classification mode (ChoiceEvaluator): use boolean
                is_correct = bool(pass_value)
                if is_correct:
                    correct += 1
                    total_score += 1.0

            # Update category stats
            if category not in category_stats:
                category_stats[category] = {"total": 0, "total_score": 0.0}
                if not use_scoring_mode:
                    category_stats[category]["correct"] = 0

            category_stats[category]["total"] += 1
            category_stats[category]["total_score"] += (
                float(pass_value) if use_scoring_mode else (1.0 if pass_value else 0.0)
            )

            if not use_scoring_mode and pass_value:
                category_stats[category]["correct"] += 1

            # Update subcategory stats
            if sub_category not in subcategory_stats:
                subcategory_stats[sub_category] = {"total": 0, "total_score": 0.0}
                if not use_scoring_mode:
                    subcategory_stats[sub_category]["correct"] = 0

            subcategory_stats[sub_category]["total"] += 1
            subcategory_stats[sub_category]["total_score"] += (
                float(pass_value) if use_scoring_mode else (1.0 if pass_value else 0.0)
            )

            if not use_scoring_mode and pass_value:
                subcategory_stats[sub_category]["correct"] += 1

        # Calculate overall metrics
        overall_avg_score = (total_score / total) if total > 0 else 0.0

        # Build overall metrics based on mode
        overall_metrics = {
            "avg_score": overall_avg_score,
            "total": total,
            "no_prediction_count": no_prediction,
        }

        # Only include accuracy/correct for classification mode
        if not use_scoring_mode:
            overall_accuracy = (correct / total * 100) if total > 0 else 0
            overall_metrics["accuracy"] = overall_accuracy
            overall_metrics["correct"] = correct

        # Calculate category metrics
        for stats in category_stats.values():
            stats["avg_score"] = (
                (stats["total_score"] / stats["total"]) if stats["total"] > 0 else 0.0
            )
            if not use_scoring_mode:
                stats["accuracy"] = (
                    (stats["correct"] / stats["total"] * 100)
                    if stats["total"] > 0
                    else 0
                )

        # Calculate subcategory metrics
        for stats in subcategory_stats.values():
            stats["avg_score"] = (
                (stats["total_score"] / stats["total"]) if stats["total"] > 0 else 0.0
            )
            if not use_scoring_mode:
                stats["accuracy"] = (
                    (stats["correct"] / stats["total"] * 100)
                    if stats["total"] > 0
                    else 0
                )

        return {
            "overall": overall_metrics,
            "category_metrics": category_stats,
            "subcategory_metrics": subcategory_stats,
        }

    def _print_results(self, metrics: Dict):
        """Print evaluation results."""
        if not metrics:
            return

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        # Overall metrics - prioritize avg_score display
        if "overall" in metrics:
            overall = metrics["overall"]

            # Display avg_score first (for Generation/Scoring mode)
            if "avg_score" in overall:
                print(
                    f"Overall Average Score: {overall['avg_score']:.3f} (Total: {overall['total']})"
                )

            # Display accuracy if available (for Classification mode)
            if "accuracy" in overall and "correct" in overall:
                print(
                    f"Overall Accuracy: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})"
                )

        # Category-wise metrics - prioritize avg_score display
        if "category_metrics" in metrics:
            print("\nCategory-wise Metrics:")
            for category, stats in metrics["category_metrics"].items():
                # Prioritize avg_score display
                if "avg_score" in stats:
                    score_str = f"Avg Score: {stats['avg_score']:.3f}"
                    if "accuracy" in stats and "correct" in stats:
                        # Show both for classification mode
                        accuracy_str = f"{stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})"
                        print(f"  {category}: {accuracy_str}, {score_str}")
                    else:
                        # Show only score for scoring mode
                        print(f"  {category}: {score_str} ({stats['total']} items)")

        # Sub-category-wise metrics - prioritize avg_score display
        if "subcategory_metrics" in metrics:
            print("\nSub-category-wise Metrics:")
            for subcat, stats in metrics["subcategory_metrics"].items():
                # Prioritize avg_score display
                if "avg_score" in stats:
                    score_str = f"Avg Score: {stats['avg_score']:.3f}"
                    if "accuracy" in stats and "correct" in stats:
                        # Show both for classification mode
                        accuracy_str = f"{stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})"
                        print(f"  {subcat}: {accuracy_str}, {score_str}")
                    else:
                        # Show only score for scoring mode
                        print(f"  {subcat}: {score_str} ({stats['total']} items)")

        if "overall" in metrics and metrics["overall"]["no_prediction_count"] > 0:
            print(
                f"\n⚠️  No prediction count: {metrics['overall']['no_prediction_count']}"
            )

        print("=" * 60)
