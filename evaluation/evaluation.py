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

def evaluate_model_performance(input_csv_path, output_csv_path=None):
    df = pd.read_csv(input_csv_path)

    predictions = df["inference_message"].astype(str).tolist()
    references = df["expected_message"].astype(str).tolist()

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
    print("\n🔍 Overall Model Evaluation:")
    print(f"BLEU-1:               {bleu_score1:.4f}")
    print(f"BLEU-2:               {bleu_score2:.4f}")
    print(f"BLEU-3:               {bleu_score3:.4f}")
    print(f"BLEU-4:               {bleu_score4:.5f}")
    print(f"METEOR:             {meteor_score:.4f}")
    print(f"ROUGE-1 F1:         {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-L F1:         {rouge_scores['rougeL']:.4f}")
    print(f"BERTScore (F1):     {bertscore_f1:.4f}")

    # Save results to CSV if output path is provided
    if output_csv_path:
        results = {
            "BLEU-1": bleu_score1 * 100,
            "BLEU-2": bleu_score2 * 100,
            "BLEU-3": bleu_score3* 100,
            "BLEU-4": bleu_score4* 100,
            "METEOR": meteor_score* 100,
            "ROUGE-1 F1": rouge_scores["rouge1"]* 100,
            "ROUGE-L F1": rouge_scores["rougeL"]* 100,
            "BERTScore (F1)": bertscore_f1* 100,
        }
        results_df = pd.DataFrame([results])
        results_df.to_csv(output_csv_path, index=False)
        print(f"Results saved to {output_csv_path}")
# Example usage
input_csv_path = "dataset/samples/evaluation_results/java_evaluation_results_gpt-3.5-turbo.csv"
output_csv_path = "dataset/samples/evaluation_results/java_evaluation_results_gpt-3.5-turbo_summary.csv"
evaluate_model_performance(input_csv_path, output_csv_path)


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
