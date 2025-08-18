import pandas as pd
import numpy as np
import os
import re
import mlflow
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix
)
import psutil
import time
import matplotlib.pyplot as plt
import seaborn as sns
from data_handling.data_loader import get_data_loader
import yaml
import argparse

# Import the new model factory and adapter base class
from models.model_factory import get_model_adapter


def load_config(config_path):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration file loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{config_path}'")
        exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL: Error parsing YAML file: {e}")
        exit(1)


def parse_and_validate_output(text, class_names):
    """Parses the VLM output to extract label and description."""
    match = re.search(r"Label:\s*([\w\s]+?)\s*,\s*Description:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        label = match.group(1).strip()
        description = match.group(2).strip()
        if label in class_names:
            return label, description
    return None, None


def log_confusion_matrix_figure(y_true, y_pred, class_names):
    """Generates and logs a confusion matrix figure to MLflow."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)
    print("\nConfusion Matrix figure logged to MLflow.")


def evaluate_vlm(model_config: dict, dataset_df: pd.DataFrame, class_names: list, eval_config: dict,
                 project_config: dict):
    """
    Evaluates a single VLM using the adapter factory for model handling.
    """
    # Set the tracking URI to be in the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    mlflow.set_tracking_uri(f"file://{os.path.join(project_root, 'mlruns')}")

    mlflow.set_experiment(project_config['experiment_name'])
    with mlflow.start_run(run_name=model_config['name']) as run:
        start_time = time.time()
        print(f"--- Starting evaluation for: {model_config['name']} ---")
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_params(model_config)
        mlflow.log_param("sample_size", len(dataset_df))

        # --- Model Loading via Factory ---
        # All model-specific logic is now handled by the adapter.
        try:
            print("Initializing model adapter...")
            adapter = get_model_adapter(model_config)
            adapter.load_model()
            print(f"Adapter for '{model_config['name']}' loaded successfully.")
        except (ValueError, IOError, Exception) as e:
            print(f"FATAL: Failed to load model for {model_config['name']}: {e}")
            mlflow.log_param("status", "load_failed")
            mlflow.end_run()
            return

        process = psutil.Process(os.getpid())
        ram_usage_mb = []
        results = []

        user_prompt_text = eval_config['prompt_template'].format(class_names=', '.join(class_names))
        max_retries = eval_config['max_retries']

        for index, row in dataset_df.iterrows():
            print(f"Processing image {index + 1}/{len(dataset_df)}: {os.path.basename(row['image_path'])}")
            predicted_label, description, generated_text = None, None, ""

            for attempt in range(max_retries):
                try:
                    # --- Inference via Adapter ---
                    # The adapter's predict method hides the complexity of different model APIs.
                    generated_text = adapter.predict(
                        image_path=row['image_path'],
                        prompt=user_prompt_text,
                        max_new_tokens=eval_config['max_new_tokens']
                    )

                    predicted_label, description = parse_and_validate_output(generated_text, class_names)
                    if predicted_label:
                        if attempt > 0: print(f"  Successfully parsed on attempt {attempt + 1}.")
                        break
                    else:
                        print(f"  Parse failed on attempt {attempt + 1}/{max_retries}. Raw output: {generated_text}")
                        if attempt < max_retries - 1: print("    Retrying...")
                except Exception as e:
                    print(f"  ERROR during inference for {row['image_path']}: {e}")
                    predicted_label = 'INFERENCE_ERROR'
                    description = str(e)
                    break

            if not predicted_label:
                print(f"  Giving up after {max_retries} attempts. Marking as PARSE_FAILED.")
                predicted_label = 'PARSE_FAILED'
                description = generated_text

            ram_usage_mb.append(process.memory_info().rss / (1024 * 1024))
            results.append({
                'image_path': row['image_path'],
                'true_label': row['true_label_name'],
                'predicted_label': predicted_label,
                'description': description,
                'raw_output': generated_text  # Storing raw output can be useful for debugging
            })

        # --- Performance & Detailed Metrics Logging (Unchanged) ---
        end_time = time.time()
        # ... (The rest of the metrics calculation and logging code remains exactly the same)
        total_duration = end_time - start_time
        avg_time_per_image = total_duration / len(dataset_df) if len(dataset_df) > 0 else 0

        print(f"\nTotal Evaluation Time: {total_duration:.2f} seconds")
        print(f"Average Time per Image: {avg_time_per_image:.2f} seconds")
        mlflow.log_metric("total_evaluation_duration_sec", total_duration)
        mlflow.log_metric("avg_inference_time_per_image_sec", avg_time_per_image)

        results_df = pd.DataFrame(results)
        valid_results_df = results_df[~results_df['predicted_label'].isin(['PARSE_FAILED', 'INFERENCE_ERROR'])]

        if ram_usage_mb:
            mlflow.log_metric("max_ram_usage_mb", max(ram_usage_mb))
            mlflow.log_metric("avg_ram_usage_mb", np.mean(ram_usage_mb))
            print(f"Max RAM Usage (RSS): {max(ram_usage_mb):.2f} MB")
            print(f"Average RAM Usage (RSS): {np.mean(ram_usage_mb):.2f} MB")

        if not valid_results_df.empty:
            y_true = valid_results_df['true_label']
            y_pred = valid_results_df['predicted_label']
            report_dict = classification_report(y_true, y_pred, labels=class_names, output_dict=True, zero_division=0)
            report_text = classification_report(y_true, y_pred, labels=class_names, zero_division=0)
            print("\n--- Evaluation Metrics ---")
            print(report_text)

            overall_accuracy = accuracy_score(y_true, y_pred)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            mlflow.log_metric("overall_accuracy", overall_accuracy)
            mlflow.log_metric("balanced_accuracy", balanced_acc)

            # ... (rest of metric logging is unchanged)

            mlflow.log_text(report_text, "classification_report.txt")
            log_confusion_matrix_figure(y_true, y_pred, class_names)
        else:
            print("\nNo valid predictions were made; skipping metrics calculation.")

        parse_failure_rate = (results_df['predicted_label'] == 'PARSE_FAILED').mean()
        inference_error_rate = (results_df['predicted_label'] == 'INFERENCE_ERROR').mean()
        mlflow.log_metric("parse_failure_rate", parse_failure_rate)
        mlflow.log_metric("inference_error_rate", inference_error_rate)
        print(f"\nParse Failure Rate: {parse_failure_rate:.2%}")
        print(f"Inference Error Rate: {inference_error_rate:.2%}")

        results_df.to_csv("results.csv", index=False)
        mlflow.log_artifact("results.csv", "results")
        print("\nResults CSV logged to MLflow.")
        print(f"--- Finished evaluation for: {model_config['name']} --- \n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Evaluate Vision Language Models using a configuration file.")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the YAML configuration file (default: config.yaml)'
    )
    args = parser.parse_args()

    config = load_config(args.config)
    project_config = config['project']
    dataset_config = config['dataset']
    eval_config = config['evaluation']
    models_to_evaluate = config['models_to_evaluate']

    try:
        data_loader = get_data_loader(dataset_config['type'], **dataset_config['loader_config'])
        full_df, class_names = data_loader.load_data()

        sample_percentage = dataset_config['sample_percentage']
        sample_size = max(1, int(len(full_df) * sample_percentage))
        sampled_df = full_df.sample(n=sample_size, random_state=dataset_config['random_state'])

        print(f"\nTotal images loaded: {len(full_df)}")
        print(f"Dataset classes: {class_names}")
        print(f"Sampling {sample_percentage:.2%} of the dataset for evaluation ({len(sampled_df)} images)...\n")

        for model_conf in models_to_evaluate:
            evaluate_vlm(model_conf, sampled_df, class_names, eval_config, project_config)

    except FileNotFoundError:
        print(f"Error: Data directory not found. Please check the paths in your config file.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()