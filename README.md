# SentimentScope: IMDB Sentiment Analysis with Transformers

## Description
An end-to-end sentiment analysis pipeline that trains a custom causal Transformer architecture (`DemoGPT`) from scratch using PyTorch and Hugging Face's `bert-base-uncased` subword tokenizer on the IMDB movie reviews dataset. 

The project adapts generative transformer blocks (multi-head causal self-attention, layer normalization, and feed-forward networks) for binary text classification through custom sequence pooling and a linear output head.

### Key Highlights
- **Architecture:** Custom multi-layer causal Transformer (`DemoGPT`) built in PyTorch
- **Tokenization:** Subword tokenization and fixed-sequence batching using Hugging Face's `AutoTokenizer`
- **Pipeline:** Custom `Dataset` and `DataLoader` pipelines with dynamic batching and shuffling
- **Optimization:** Trained using AdamW with gradient clipping and cross-entropy loss, exceeding the project benchmark with >77% accuracy
