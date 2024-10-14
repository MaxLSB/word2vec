# Word2Vec & Classification

# Abstract
This project is about the implementation of Word2Vec from scratch in PyTorch, as introduced by Mikolov et al. (2013) in the **Large Language Models** course at Université Paris Dauphine. Word2Vec is a neural network model that generates dense vector representations of words, capturing semantic relationships in text. This implementation explores key techniques of the architecture, including the use of positive and negative contexts, data preprocessing, the model class, and the training process. We then use the pre-trained embeddings as a starting point for our classifier and compare the impact of various Word2Vec parameters on the model's performance.

_The report is also available._

# Introduction
The Word2Vec algorithm samples a word from a dataset and identifies its surrounding words within a specified window radius $R$ to create positive context examples. Additionally, the model utilizes negative sampling, where random words from the vocabulary are selected as negative examples. This implementation computes similarity scores between the target word and context words using the sigmoid function, with the objective of maximizing similarity for positive contexts and minimizing it for negative contexts. This process enables the model to learn effective word embeddings that capture the underlying semantic structure of the text. We will use a pre-built BERT tokenizer and Hugging Face's sentiment analysis dataset 'imdb'.

# Installation
- Install the requiered packages.
- Modify the hyperparameters in the main functions of ```word2vec.py``` or ```classification.py``` as you please.
- If you want to use another dataset consider making serveral changes in the code.
