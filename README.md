# NLP Course Projects

This repository contains three projects developed for a Natural Language Processing course. The focus of these implementations is on the **conceptual understanding** of the algorithms through **from-scratch implementations using NumPy**, rather than performance or production-readiness.

The implemented algorithms:
1. Byte Pair Encoding (BPE)
2. Skip-gram with Negative Sampling
3. Sentiment Analysis

Code is built to be run in a linux environment, but there should be no problem running it in a windows environment with the proper dependencies installed. It also is made to be self-explanatory, with comments and docstrings explaining the code, so head over to the code to understand it better (mixed comments in spanish and english though).

---

## Project 1: Byte Pair Encoding (BPE)
**Folder:** `P1_Bpe`

### Overview
A basic implementation of the Byte Pair Encoding (BPE) tokenization algorithm. It learns a vocabulary by iteratively merging the most frequent adjacent characters (or bytes) in a text corpus.

### Inner Workings
- Starts with a vocabulary of individual bytes.
- Iteratively counts pairs of adjacent tokens.
- Merges the most frequent observations.
- Updates the text and repeats until the desired vocabulary size is reached.

> [!NOTE] 
> **Efficiency Note**
> The current implementation is naive and recalculates pair frequencies from scratch in every iteration of the merge loop. A more efficient implementation would update these counts incrementally or use a dedicated data structure (like a linked list) to track adjacent pairs, significantly reducing the algorithmic complexity.

### Data Dependencies
- Text corpus for training (e.g., `tiny_cc_news.txt`, `rajoy_corpus.txt`).

---

## Project 2: Skip-gram with Negative Sampling
**Folder:** `P2_Skipgram`

### Overview
An implementation of the Word2Vec Skip-gram model to generate word embeddings. This project constructs vector representations of words such that words appearing in similar contexts have similar vector representations.

### Inner Workings
- **Tokenization:** Uses the BPE tokenizer from Project 1.
- **Model:** A custom `Trainer` class that implements the Skip-gram architecture.
- **Optimization:** Manual implementation of Stochastic Gradient Descent (SGD) with linear learning rate decay.
- **Features:** 
    - **Subsampling:** Frequent words are discarded with a probability related to their frequency.
    - **Negative Sampling:** Approximates the Softmax function by sampling "negative" examples (words not in context) using a smoothed frequency distribution ($f^{0.75}$).
    - **Dynamic Window:** Context window size is sampled randomly during training in order to improve the model's ability to capture both short-range and long-range dependencies.

### Data Dependencies
- Pre-computed encoded tokens (e.g., `encoded_tokens_1000.txt`) or a raw text corpus processed by the BPE model.
- BPE model from Project 1.

---

## Project 3: Sentiment Analysis
**Folder:** `P3_SentimentAnalysis`

### Overview
A binary classification project to determine the sentiment (positive/negative) of text samples from a small toy corpus. It utilizes the embeddings generated in Project 2.

### Inner Workings
- **Classifier:** A `LogisticRegression` class implemented from scratch using NumPy.
    - Manual implementation of the `forward` (sigmoid activation) and `backward` (gradient computation) passes.
    - Implements and uses Binary Cross Entropy loss.
- **Input:** Aggregated word embeddings (averaging the vectors of tokens in a sentence) are used as input features for the classifier.
- **Extra:** The folder also contains `finetune.py`, which demonstrates how to fine-tune a pre-trained BERT model using the Hugging Face `transformers` library for comparison (this part is not made from scratch).

### Data Dependencies
- Labeled training and test datasets (TSV format): `train.tsv`, `test.tsv`.
- Word embeddings from Project 2 (`embeddings.txt`).
- BPE model from Project 1.
