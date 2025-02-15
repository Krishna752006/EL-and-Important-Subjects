'''You are tasked with creating a pipeline for textual data preprocessing, involving multiple steps for cleaning, transforming, and analyzing text. Below are the steps: like tokenization and sentence splitting.

Import Libraries Import essential libraries like torch, string, and re for processing text and handling operations
Text Loading and Initialization Define the input text (e.g., a paragraph) for preprocessing.

Tokenization and Lowercasing Break the text into individual tokens (words) while removing punctuation and whitespace. Convert all tokens to lowercase for uniformity.

Sentence Splitting and Counting Divide the paragraph into sentences using punctuation markers and count the number of non-empty sentences.

Identifying Uppercase Words and Extremes Extract words written in uppercase and find the longest and shortest tokens.

Numerical Tokens Find tokens that are numbers and calculate the count.

Filtering Long Tokens Identify tokens with lengths greater than 4 characters and count them.

Synonym Replacement Replace tokens with their synonyms using a predefined dictionary, simplifying or enhancing semantic meaning.

Handling Negations Expand contractions (e.g., "don't" to "do not") and tokenize the text, ensuring explicit handling of negations.

Lemmatization Convert tokens to their base forms (e.g., "loved" to "love") using a predefined mapping, ensuring consistency in word representation. '''




import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import re
import string

paragraph = """
Machine Learning (ML) is a fascinating field that enables COMPUTERS to LEARN from data and make predictions.
It uses ALGORITHMS like decision trees, neural networks, and k-means clustering. Over 50% of industries now use ML in 
various applications.ML is transforming how we live and work.By 2030, this TECHNOLOGY will shape many fields, including 
TRANSPORT and EDUCATION. Are we ready for ML?!"""

def tokenize(text):
    tokens = []
    word = ''

    for char in text:
        if char in string.whitespace or char in string.punctuation:
            if word: 
                tokens.append(word)

                word = ''
        else:
            word += char
    if word:
        tokens.append(word)

    return tokens

tokens = tokenize(paragraph)
lower_tokens = [word.lower() for word in tokens]

def split_to_sentences(paragraph):
    sentences=re.split(r'[.!]',paragraph)
    filtered_sentences=[s.strip() for s in sentences if s.strip()!='']
    sentence_count=len(filtered_sentences)
    return filtered_sentences,sentence_count
sentences,count=split_to_sentences(paragraph)

def extract_uppercase_words(tokens):
    uppercase_words=[word for word in tokens if word.isupper()]
    return uppercase_words
uppercase_words=extract_uppercase_words(tokens)

longest_word = max(tokens, key=len)
shortest_word = min(tokens, key=len)

numeric_tokens = [word for word in tokens if word.isdigit()]
numeric_token_count = len(numeric_tokens)

long_tokens = [word for word in tokens if len(word) > 4]
long_tokens_count = len(long_tokens)

def replace_synonyms(tokens):
    synonyms = {
        'fascinating': 'interesting',
        'computers': 'machines',
        'data': 'information',
        'learning': 'studying'
    }
    return [synonyms.get(token, token) for token in tokens]

synonym_replaced_tokens = replace_synonyms(lower_tokens)