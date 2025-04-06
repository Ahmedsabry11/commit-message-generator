# 🧠 Commit Message Generator

Generate high-quality, convention-compliant commit messages from code diffs using Large Language Models (LLMs). This project explores prompt engineering strategies and evaluates how well LLMs perform on real-world commits using datasets like **CommitBench** and **MCMD**.

---

## 📁 Project Structure

```
commit-message-generator/
├── main.py                   # Entry point for testing prompts
├── prompts/prompt.py         # Prompt engineering templates
├── models/openai.py    ``````# GPT-3.5/4 API wrapper
├── dataset/sample.py         # Sample diffs or dataset access
├── dataset/commitbench.csv   # csv of commitbench dataset
├── evaluation/scorer.py      # Evaluation metrics (optional)
├── .env                      # OpenAI API key
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/commit-message-generator.git
cd commit-message-generator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure OpenAI API

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your-openai-api-key-here
```
---

## 📦 Dataset Setup

This project supports two major datasets: **CommitBench** and **MCMD**.

### ✅ CommitBench

1. Download dataset from [CommitBench Dataset](https://zenodo.org/records/10497442) in dataset folder
---

### ✅ MCMD (Massive Code Message Dataset)

1. Download dataset from [MCMD Dataset](https://zenodo.org/record/7196966#.Y0juJHZBxmM) in dataset folder
---

## 🚀 Running the Generator

To test the generator with sample prompts and diffs:

```bash
python main.py
```

You can customize:
- Prompts in `prompts/prompt.py`
- Input diffs in `dataset/sample.py`

---

## 🧪 Evaluation (Optional)

Coming soon or in development:

- Add BLEU, ROUGE, or METEOR score comparison in `evaluation/scorer.py`
- Compare LLM output with human commit messages
- Use human ratings for clarity, tone, length, and usefulness

---

## ✨ Prompt Engineering Strategies

Prompts can dramatically affect the output quality. We support:

- **Zero-shot prompts**: "Summarize this diff"
- **Conventional commits**: Using `feat:`, `fix:`, `refactor:`, etc.
- **Imperative tone prompts**
- **Minimalist messages** (<=8 words)
- **Motivation + change**: Explain why + what

Edit and experiment with `prompts/prompt.py`.
---

## 🧠 Ideas to implement

- ✅ Streamlit UI for testing prompts
- ✅ Multiple model comparison (GPT-3.5, GPT-4, StarCoder, etc.)
- 🧪 Fine-tuning small models (CodeT5+)
- 📊 Logging and comparing prompt results in CSV
- 🧰 Integration into Git commit workflow

---

## 📚 References

- 🔗 [LLM4CMG](https://github.com/wuyifan18/LLM4CMG/tree/main)
- 🔗 [commitbench](https://github.com/maxscha/commitbench)
- 🔗 [OpenAI GPT API Docs](https://platform.openai.com/docs)
- 📄 BLEU/ROUGE from NLTK or HuggngFace `evaluate`

---

## 🛡 License

MIT License — free to use, improve, and distribute.

---
