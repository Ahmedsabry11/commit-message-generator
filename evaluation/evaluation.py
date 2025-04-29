import evaluate

# Load metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
# bertscore_metric = evaluate.load("bertscore")

def evaluate_outputs(reference, generated):
    references = [reference]
    predictions = [generated]

    # BLEU
    bleu1 = bleu_metric.compute(predictions=predictions,
                         references=[[ref] for ref in references],
                         max_order=1)["bleu"]

    bleu2 = bleu_metric.compute(predictions=predictions,
                         references=[[ref] for ref in references],
                         max_order=2)["bleu"]

    bleu3 = bleu_metric.compute(predictions=predictions,
                         references=[[ref] for ref in references],
                         max_order=3)["bleu"]

    bleu4 = bleu_metric.compute(predictions=predictions,
                         references=[[ref] for ref in references],
                         max_order=4)["bleu"]

    # bleu_result = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    
    # ROUGE
    rouge_result = rouge_metric.compute(predictions=predictions, references=references)

    # BERTScore
    # bert_result = bertscore_metric.compute(predictions=predictions, references=references, lang="en")

    return {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
        "ROUGE-1": rouge_result["rouge1"],
        "ROUGE-2": rouge_result["rouge2"],
        "ROUGE-L": rouge_result["rougeL"],
        # "BERTScore (P)": sum(bert_result["precision"]) / len(bert_result["precision"]),
        # "BERTScore (R)": sum(bert_result["recall"]) / len(bert_result["recall"]),
        # "BERTScore (F1)": sum(bert_result["f1"]) / len(bert_result["f1"]),
    }

if __name__ == "__main__":
    # Example usage
    reference = "Fix bug in user authentication flow"
    generated = "Fixed a bug in the user authentication process."

    results = evaluate_outputs(reference, generated)
    print("Evaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
