import evaluate

# Load metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
# bertscore_metric = evaluate.load("bertscore")

def evaluate_outputs(reference, generated):
    references = [reference]
    predictions = [generated]

    # BLEU
    bleu_result = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    
    # ROUGE
    rouge_result = rouge_metric.compute(predictions=predictions, references=references)

    # BERTScore
    # bert_result = bertscore_metric.compute(predictions=predictions, references=references, lang="en")

    return {
        "BLEU": bleu_result["bleu"],
        "ROUGE-1": rouge_result["rouge1"],
        "ROUGE-2": rouge_result["rouge2"],
        "ROUGE-L": rouge_result["rougeL"],
        # "BERTScore (P)": sum(bert_result["precision"]) / len(bert_result["precision"]),
        # "BERTScore (R)": sum(bert_result["recall"]) / len(bert_result["recall"]),
        # "BERTScore (F1)": sum(bert_result["f1"]) / len(bert_result["f1"]),
    }
