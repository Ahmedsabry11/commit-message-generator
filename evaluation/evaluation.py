import pandas as pd
import os
os.environ["USE_TF"] = "0"
import evaluate
from tqdm import tqdm

# Load metrics
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def evaluate_model_performance(input_csv_path, model_name, language, prompt_style,results_list):
    df = pd.read_csv(input_csv_path)

    predictions = df["inference_message"].astype(str).tolist()
    references = df["expected_message"].astype(str).tolist()

    print(f"\n🔎 Evaluating {model_name} ({language}) ({prompt_style}....")
    print(f"Number of samples: {len(predictions)}")

    print("Computing BLEU...")
    bleu_score1 = bleu.compute(predictions=predictions, references=[[r] for r in references],max_order=1)["bleu"]
    bleu_score2 = bleu.compute(predictions=predictions, references=[[r] for r in references],max_order=2)["bleu"]
    bleu_score3 = bleu.compute(predictions=predictions, references=[[r] for r in references],max_order=3)["bleu"]
    bleu_score4 = bleu.compute(predictions=predictions, references=[[r] for r in references],max_order=4)["bleu"]

    print("Computing METEOR...")
    meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]

    print("Computing ROUGE...")
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    print("Computing BERTScore...")
    bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")
    bertscore_f1 = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])

    # Print overall results
    print("=" * 100)
    print(f"🔍 Model: {model_name} ({language}) ({prompt_style})")
    print("\n🔍 Overall Model Evaluation:")
    print(f"BLEU-1:               {bleu_score1:.4f}")
    print(f"BLEU-2:               {bleu_score2:.4f}")
    print(f"BLEU-3:               {bleu_score3:.4f}")
    print(f"BLEU-4:               {bleu_score4:.5f}")
    print(f"METEOR:             {meteor_score:.4f}")
    print(f"ROUGE-1 F1:         {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-L F1:         {rouge_scores['rougeL']:.4f}")
    print(f"BERTScore (F1):     {bertscore_f1:.4f}")
    print("=" * 100)

    # Append results to the DataFrame
    results_list.append({
        "Model": model_name,
        "Language": language,
        "prompt_style": prompt_style,
        "BLEU-1": bleu_score1 * 100,
        "BLEU-2": bleu_score2 * 100,
        "BLEU-3": bleu_score3 * 100,
        "BLEU-4": bleu_score4 * 100,
        "METEOR": meteor_score * 100,
        "ROUGE-1 F1": rouge_scores["rouge1"] * 100,
        "ROUGE-L F1": rouge_scores["rougeL"] * 100,
        "BERTScore (F1)": bertscore_f1 * 100,
    })

    # # Save results to CSV if output path is provided
    # if output_csv_path:
    #     results = {
    #         "BLEU-1": bleu_score1 * 100,
    #         "BLEU-2": bleu_score2 * 100,
    #         "BLEU-3": bleu_score3* 100,
    #         "BLEU-4": bleu_score4* 100,
    #         "METEOR": meteor_score* 100,
    #         "ROUGE-1 F1": rouge_scores["rouge1"]* 100,
    #         "ROUGE-L F1": rouge_scores["rougeL"]* 100,
    #         "BERTScore (F1)": bertscore_f1* 100,
    #     }
    #     results_df = pd.DataFrame([results])
    #     results_df.to_csv(output_csv_path, index=False)
    #     print(f"Results saved to {output_csv_path}")


if __name__ == "__main__":
    print("Running evaluation on the dataset...")
        # List of models and languages
    evaluations = [
        # Java
        ("dataset/samples/evaluation_results/java_evaluation_results_gemini-2.0-flash_feature.csv", "Gemini 2.0 Flash", "Java", "feature"),
        ("dataset/samples/evaluation_results/java_evaluation_results_gemini-2.0-flash_zero_shot.csv", "Gemini 2.0 Flash Zero-Shot", "Java", "zero_shot"),

        ("dataset/samples/evaluation_results/java_evaluation_results_gpt-3.5-turbo_feature.csv", "GPT-3.5 Turbo", "Java", "feature"),
        ("dataset/samples/evaluation_results/java_evaluation_results_gpt-3.5-turbo_zero_shot.csv", "GPT-3.5 Turbo Zero-Shot", "Java", "zero_shot"),
        ("dataset/samples/evaluation_results/java_evaluation_results_gpt-4.1-mini_feature.csv", "GPT-4.1 Mini", "Java", "feature"),
        ("dataset/samples/evaluation_results/java_evaluation_results_gpt-4.1-mini_zero_shot.csv", "GPT-4.1 Mini Zero-Shot", "Java", "zero_shot"),
        
        # Python
        ("dataset/samples/evaluation_results/py_evaluation_results_gemini-2.0-flash_feature.csv", "Gemini 2.0 Flash", "Python", "feature"),
        ("dataset/samples/evaluation_results/py_evaluation_results_gemini-2.0-flash_zero_shot.csv", "Gemini 2.0 Flash Zero-Shot", "Python", "zero_shot"),

        ("dataset/samples/evaluation_results/py_evaluation_results_gpt-3.5-turbo_feature.csv", "GPT-3.5 Turbo", "Python", "feature"),
        ("dataset/samples/evaluation_results/py_evaluation_results_gpt-3.5-turbo_zero_shot.csv", "GPT-3.5 Turbo Zero-Shot", "Python", "zero_shot"),
        ("dataset/samples/evaluation_results/py_evaluation_results_gpt-4.1-mini_feature.csv", "GPT-4.1 Mini", "Python", "feature"),
        ("dataset/samples/evaluation_results/py_evaluation_results_gpt-4.1-mini_zero_shot.csv", "GPT-4.1 Mini Zero-Shot", "Python", "zero_shot"),

        # JS
        ("dataset/samples/evaluation_results/js_evaluation_results_gemini-2.0-flash_feature.csv", "Gemini 2.0 Flash", "JavaScript", "feature"),
        ("dataset/samples/evaluation_results/js_evaluation_results_gemini-2.0-flash_zero_shot.csv", "Gemini 2.0 Flash Zero-Shot", "JavaScript", "zero_shot"),

        ("dataset/samples/evaluation_results/js_evaluation_results_gpt-3.5-turbo_feature.csv", "GPT-3.5 Turbo", "JavaScript", "feature"),
        ("dataset/samples/evaluation_results/js_evaluation_results_gpt-3.5-turbo_zero_shot.csv", "GPT-3.5 Turbo Zero-Shot", "JavaScript", "zero_shot"),
        ("dataset/samples/evaluation_results/js_evaluation_results_gpt-4.1-mini_feature.csv", "GPT-4.1 Mini", "JavaScript", "feature"),
        ("dataset/samples/evaluation_results/js_evaluation_results_gpt-4.1-mini_zero_shot.csv", "GPT-4.1 Mini Zero-Shot", "JavaScript", "zero_shot"),

        # PHP
        ("dataset/samples/evaluation_results/php_evaluation_results_gemini-2.0-flash_feature.csv", "Gemini 2.0 Flash", "PHP", "feature"),
        ("dataset/samples/evaluation_results/php_evaluation_results_gemini-2.0-flash_zero_shot.csv", "Gemini 2.0 Flash Zero-Shot", "PHP", "zero_shot"),

        ("dataset/samples/evaluation_results/php_evaluation_results_gpt-3.5-turbo_feature.csv", "GPT-3.5 Turbo", "PHP", "feature"),
        ("dataset/samples/evaluation_results/php_evaluation_results_gpt-3.5-turbo_zero_shot.csv", "GPT-3.5 Turbo Zero-Shot", "PHP", "zero_shot"),
        ("dataset/samples/evaluation_results/php_evaluation_results_gpt-4.1-mini_feature.csv", "GPT-4.1 Mini", "PHP", "feature"),
        ("dataset/samples/evaluation_results/php_evaluation_results_gpt-4.1-mini_zero_shot.csv", "GPT-4.1 Mini Zero-Shot", "PHP", "zero_shot"),
    ]


    results_list = []
    for input_path, model_name, language, prompt_name in evaluations:
        # Evaluate model performance
        evaluate_model_performance(input_path, model_name, language, prompt_name, results_list)

    # Save results to CSV
    results_df = pd.DataFrame(results_list)
    output_csv_path = "dataset/samples/evaluation_results/summary_results.csv"
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")



# import evaluate

# # Load metrics
# bleu_metric = evaluate.load("bleu")
# rouge_metric = evaluate.load("rouge")
# # bertscore_metric = evaluate.load("bertscore")

# def evaluate_outputs(reference, generated):
#     references = [reference]
#     predictions = [generated]

#     # BLEU
#     bleu1 = bleu_metric.compute(predictions=predictions,
#                          references=[[ref] for ref in references],
#                          max_order=1)["bleu"]

#     bleu2 = bleu_metric.compute(predictions=predictions,
#                          references=[[ref] for ref in references],
#                          max_order=2)["bleu"]

#     bleu3 = bleu_metric.compute(predictions=predictions,
#                          references=[[ref] for ref in references],
#                          max_order=3)["bleu"]

#     bleu4 = bleu_metric.compute(predictions=predictions,
#                          references=[[ref] for ref in references],
#                          max_order=4)["bleu"]

#     # bleu_result = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    
#     # ROUGE
#     rouge_result = rouge_metric.compute(predictions=predictions, references=references)

#     # BERTScore
#     # bert_result = bertscore_metric.compute(predictions=predictions, references=references, lang="en")

#     return {
#         "BLEU-1": bleu1,
#         "BLEU-2": bleu2,
#         "BLEU-3": bleu3,
#         "BLEU-4": bleu4,
#         "ROUGE-1": rouge_result["rouge1"],
#         "ROUGE-2": rouge_result["rouge2"],
#         "ROUGE-L": rouge_result["rougeL"],
#         # "BERTScore (P)": sum(bert_result["precision"]) / len(bert_result["precision"]),
#         # "BERTScore (R)": sum(bert_result["recall"]) / len(bert_result["recall"]),
#         # "BERTScore (F1)": sum(bert_result["f1"]) / len(bert_result["f1"]),
#     }

# if __name__ == "__main__":
#     # Example usage
#     reference = "Fix bug in user authentication flow"
#     generated = "Fixed a bug in the user authentication process."

#     results = evaluate_outputs(reference, generated)
#     print("Evaluation Results:")
#     for metric, score in results.items():
#         print(f"{metric}: {score:.4f}")
