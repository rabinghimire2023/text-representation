# SPRINT 4: Text Representations and Vector Databases

# What are Word Embeddings?

Word Embeddings are the texts converted into numbers. It is an approach for representing words and documents. 

 

### Goals of Word Embeddings

- To reduce dimensionality
- To use a word to predict the words around it.
- Inter-word semantics must be captured.

## Implementations of Word Embeddings

Word Embedding is a method of extracting features out of text so that we can input those features into a machine-learning model to work with text data. It preserves syntactical and semantic information. Methods such as the Goal of Words (BOW), CountVectorizer, and TFIDF rely on the word count in a sentence. In these algorithms, the size of the vector is the number of elements in the vocabulary. 

## One-Hot Encoding

In traditional NLP, words are often represented as one-hot vectors. Each word in the vocabulary is assigned a unique index, and the vector is as long as the vocabulary size.

Example: Consider a small vocabulary with three words: “Cat”, ”Dog”, and “bird”. The one-hot encodings for these words might look like:

- “cat” → [1,0,1]
- “dog” → [0,1,0]
- “bird” → [0,0,1]

One-hot encodings are simple but lack any inherent understanding of the relationships between words. Word embeddings, on the other hand, capture semantic similarities.

## Word Embeddings

Word embeddings represent words as dense vectors in a continuous space. These vectors are learned from large text corpora using neural networks, like Word2Vec. Word2Vec produces word vectors that encode semantic relationships. Words with similar meanings have similar vector representations.

Example: Using Word2Vec, the word vectors for “cat”, “dog”, and “bird” might look like this:

- “Cat” → [0.2,0.8,-0.3]
- “Dog” → [0.1,0.9,-0.2]
- “bird” → [0.3, 0.2,0.7]

# Different Types of Embeddings

## 1. Frequency-based Embeddinngs

Frequency-based embeddings, also known as count-based embeddings distribution embeddings, or distributional embeddings, are a class of word embeddings that represent words based on their co-occurrence patterns in a large corpus of text. These embeddings capture the statistical relationships between words by considering how often words appear together in sentences or documents. Two common techniques for creating frequency-based embeddings  are Term Frequency - Inverse Document Frequency (TF-IDF) and Latent Semantic Analysis (LSA):

### i) TF-IDF (Term Frequency - Inverse Document Frequency):

TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). It represents each word in a document as a numerical vector based on its frequency (TF) in the document and its rarity (IDF) in the entire corpus. TF-IDF assigns higher values to words that appear frequently in a specific document but infrequently in the corpus, indicating their significance to that document. The TF-IDF vectors for each word can be used as word embeddings. 

**Term Frequency:** It is the occurrence of the current word in the current sentence w.r.t. the total number of words in the current sentence.

![Screenshot 2023-10-13 093933.png](SPRINT%204%20Text%20Representations%20and%20Vector%20Databases%206567ba269f08494385734d7f61364569/Screenshot_2023-10-13_093933.png)

**Inverse Data Frequency:** It is the log of total number of words in the whole data corpus w.r.t. the total number of sentences containing the current word.

![Screenshot 2023-10-13 162659.png](SPRINT%204%20Text%20Representations%20and%20Vector%20Databases%206567ba269f08494385734d7f61364569/Screenshot_2023-10-13_162659.png)

Example: 

We have 5 sentences namely, [“this is a good phone” , “this is a bad mobile” , “she is a good cat” , “he has a bad temper” , “this mobile phone is not good”]

**Data Corpus:** [“a” , “bad” , “cat” , “good” , “has” , “he” , “is” , “mobile” , “not” , “phone” , “she” , “temper” , “this”]

**TF-IDF (“this” in sentence1):** Number of “this” word in sentence1 / total number of words in sentence1

**IDF :** log(total number of words in the whole data corpus / total number of sentences having “this” word)

**TF:** 1/5 = 0.2

**IDF: log(13/3) = 1.4663**

**********************************************************TF-IDF: 0.2 * 1.4663 = 0.3226**********************************************************

Similarly, we can find the TF-IDF for every other words.

The values in the TF-IDF vectors reflect the importance of each word within the document and the corpus.

### ii) Latent Semantic Analysis (LSA)

LSA is a dimensionality reduction technique that applies singular value decomposition (SVD) to a term-document matrix. It uncovers latent semantic structures by identifying patterns of word co-occurrence in the corpus. LSA produces a reduced-dimensional representation of words and documents in a way that captures semantic relationships between words. These reduced-dimensional representations can be used as word embeddings.

Example: LSA might identify that “cat” and “dog” are both associated with “pets” and  ”animals” in the latent space, indicating their semantic similarity. 

### iii) Count Vector

A count vector, also known as a term frequency vector, is a numerical representation of a document or a piece of text in natural language processing (NLP). It is a simple and straightforward way to represent the presence and frequency of words in a document. Count vectors are used in various NLP tasks, including text classification, information retrieval, and document analysis.

Here's how count vectors work:

1. **Tokenization:** The first step is to tokenize the text, which means splitting it into individual words or tokens. Punctuation and whitespace are typically used as delimiters.
2. **Vocabulary:** A vocabulary is created by collecting all unique words (tokens) from the entire corpus of documents. Each unique word in the vocabulary is assigned a unique index.
3. **Counting:** For each document in the corpus, a count vector is created. The vector has a length equal to the size of the vocabulary, and each element corresponds to a word in the vocabulary. The value in each element of the vector represents the number of times the corresponding word appears in the document.

Here’s a simple example to illustrate count vectors:

Consider a small corpus with two documents:

Document 1: “I love cats and dogs.”

Documents 2: “Cats are better than dogs.”

Vocabulary: [”I”, “love”, “cats”, “dogs”, “are”, “better”, “than”]

**Count vectors:** 

For Document1: [1,1,1,1,0,0,0,0]

- “I” appears once, “love” appears once, “cats” appears once, and so on.

For Document 2: [0,0,1,0,1,1,1,1]

- “cats” appears once, “dogs” appears once, “are” appears once, and so on.

Each document is now represented as a count vector, where the position of each word in the vector corresponds to its count in the document.

**Limitations of Count Vectors:**

- They can result in high-dimensional and sparse representations, especially for large vocabularies.

## iv) Bag of Words (BOW)

In BOW, the whole data corpus is used to encode our sentences. 

A "bag of words" (BoW) is a simple and fundamental technique in natural language processing (NLP) and text analysis. It's a way to represent text data as a collection of individual words without considering their order, structure, or grammar. In a BoW model, each document is represented as a "bag" or "set" of words, and it's primarily used for tasks like text classification, sentiment analysis, and information retrieval. Here's how it works:

1. **Tokenization**: The first step is to break down the text into individual words or tokens. Punctuation and whitespace are typically used as delimiters.
2. **Vocabulary Building**: Next, a vocabulary is created by collecting all the unique words from the entire corpus. Each unique word becomes a feature in the BoW model.
3. **Vectorization**: For each document, a vector is created with a length equal to the vocabulary size. Each position in the vector corresponds to a unique word, and the value represents the frequency of that word in the document. If a word is not present in the document, its count will be zero.

**Example**: Let's say we have a corpus with two simple documents:

Document 1: "I like cats."
Document 2: "I like dogs and cats."

Tokenization:
Document 1: ["I", "like", "cats"]
Document 2: ["I", "like", "dogs", "and", "cats"]

Vocabulary:
["I", "like", "cats", "dogs", "and"]

Vectorization:
Document 1: [1, 1, 1, 0, 0] (The numbers represent word counts in the same order as the vocabulary)
Document 2: [1, 1, 1, 1, 1]

In this way, each document is represented as a BoW vector, which is a numerical representation of the text. BoW is a straightforward method that doesn't capture the semantics or order of words in a document, but it can be useful for certain text analysis tasks, like spam detection or document categorization.

# 2. Prediction Based Embedding

Predictive-based embeddings are a type of word embedding model that learns to represent words or tokens in a vector space by predicting their context within a given text corpus. These embeddings capture the semantic meaning of words based on the distributional hypothesis, which suggests that words with similar meanings tend to occur in similar contexts.

## i) Word2Vec

Word2vec is a method to efficiently create word embeddings.Word2Vec creates vectors of the words that are distributed numerical representations of word features – these word features could comprise words that represent the context of the individual words present in our vocabulary.

Two different model architectures that can be used by Word2Vec to create the word embeddings are the Continuous Bag of Words (CBOW) model & the Skip-Gram model.

### CBOW

The CBOW architecture comprises a deep learning classification model in which we take in context words as input, X, and try to predict our target word, Y. The way CBOW works is that it tends to predict the probability of a word given a context. A context may be a single word or a group of words.

In this approach, the model predicts a target word based on its surrounding context words. The model takes a context window of words and tries to predict the target word in the middle. The idea is to learn the word embeddings by understanding what words are likely to occur in similar contexts.

The following steps describe how the model works:

- The context words are first passed as input to an embedding layer (initialized with some random weights) as shown in the Figure below.
- The word embeddings are then passed to a lambda layer where we average out the word embeddings.
- We then pass these embeddings to a dense SoftMax layer that predicts our target word. We match this with our target word and compute the loss and then we perform backpropagation with each epoch to update the embedding layer in the process.

We can extract the embeddings of the needed words from our embedding layer, once the training is completed.

![32272Picture3.png](SPRINT%204%20Text%20Representations%20and%20Vector%20Databases%206567ba269f08494385734d7f61364569/32272Picture3.png)

**Advantages of CBOW:**

1. Being probabilistic in nature, it is supposed to perform superior to deterministic methods(generally).
2. It is low on memory. It does not need to have huge RAM requirements like that of a co-occurrence matrix where it needs to store three huge matrices.

### Skip-gram

This method works the other way around. Given a target word, skip-gram tries to predict the words that are likely to appear in its context. This approach is often preferred when you have a large corpus of text and want to capture more detailed information about each word.

The aim of skip-gram is to predict the context given a word.

The following steps describe how the model works:

- Both the target and context word pairs are passed to individual embedding layers from which we get dense word embeddings for each of these two words.
- We then use a ‘merge layer’ to compute the dot product of these two embeddings and get the dot product value.
- This dot product value is then sent to a dense sigmoid layer that outputs either 0 or 1.
- The output is compared with the actual label and the loss is computed followed by backpropagation with each epoch to update the embedding layer in the process.

![40007Capture.png](SPRINT%204%20Text%20Representations%20and%20Vector%20Databases%206567ba269f08494385734d7f61364569/40007Capture.png)

## ii) **GloVe: Global Vectors for Word Representation**

GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

The basic idea behind the GloVe word embedding is to derive the relationship between the words from statistics. Unlike the occurrence matrix, the co-occurrence matrix tells you how often a particular word pair occurs together. Each value in the co-occurrence matrix represents a pair of words occurring together.

The advantage of GloVe is that, unlike Word2vec, GloVe does not rely just on local statistics (local context information of words), but incorporates global statistics (word co-occurrence) to obtain word vectors.

The mathematical representation for the GloVe is :

![Screenshot 2023-10-17 141259.png](SPRINT%204%20Text%20Representations%20and%20Vector%20Databases%206567ba269f08494385734d7f61364569/Screenshot_2023-10-17_141259.png)

## iii) ELMO embeddings

ELMo (Embeddings from Language Models) is a pre-trained contextual word embedding model introduced by researchers from the Allen Institute for Artificial Intelligence (AI2).

ELMo embeddings are based on deep contextualized word representations, which means that the embeddings take into account the context in which a word appears in a sentence. This allows ELMo embeddings to capture nuances and word sense disambiguation, making them highly useful in various natural language processing (NLP) tasks.

ELMo word vectors are computed on top of a two-layer bidirectional language model (biLM). This biLM model has two layers stacked together. Each layer has 2 passes — forward pass and backward pass:

![output_YyJc8E.gif](SPRINT%204%20Text%20Representations%20and%20Vector%20Databases%206567ba269f08494385734d7f61364569/output_YyJc8E.gif)

- The architecture above uses a character-level convolutional neural network (CNN) to represent words of a text string into raw word vectors
- These raw word vectors act as inputs to the first layer of biLM
- The forward pass contains information about a certain word and the context (other words) before that word
- The backward pass contains information about the word and the context after it
- This pair of information, from the forward and backward pass, forms the intermediate word vectors
- These intermediate word vectors are fed into the next layer of biLM
- The final representation (ELMo) is the weighted sum of the raw word vectors and the 2 intermediate word vectors.

### Differences between ELMO from other embeddings

ELMo (Embeddings from Language Models) differs from traditional word embeddings like Word2Vec and GloVe in several significant ways:

1. Contextual Embeddings:
    - Traditional word embeddings (e.g., Word2Vec, GloVe) assign a fixed vector representation to each word in the vocabulary. These representations are context-independent, meaning that the same word is represented by the same vector, regardless of its context in a sentence.
    - ELMo, on the other hand, provides contextual embeddings. Each word's representation varies depending on its context within the sentence. This context awareness allows ELMo to capture polysemy (multiple meanings of a word) and word sense disambiguation more effectively.
2. Deep Learning Architecture:
    - Word2Vec and GloVe are shallow models, typically based on simple neural network architectures, and they do not take into account the larger sentence context.
    - ELMo is based on deep learning architecture, specifically a bidirectional LSTM (Long Short-Term Memory) network. It considers not only the words before a target word but also the words after it, which helps in capturing the context and dependencies within a sentence.
3. Subword Information:
    - Traditional word embeddings are typically based on entire words. They don't handle out-of-vocabulary words or subword information well.
    - ELMo can generate embeddings for out-of-vocabulary words and is aware of subword information. It uses character-level CNNs (Convolutional Neural Networks) to represent subword information effectively.
4. Dynamic Feature Extraction:
    - Traditional word embeddings generate a fixed representation for each word, which might not be suitable for tasks with varying word meanings in different contexts.
    - ELMo can dynamically generate embeddings that change depending on the specific context. It computes context-sensitive representations as a linear combination of embeddings from different layers of the deep architecture.
5. Improved Performance:
    - ELMo embeddings have been shown to significantly improve the performance of various NLP tasks, such as sentiment analysis, named entity recognition, question answering, and more, compared to traditional word embeddings.
    - ELMo's ability to capture context and semantics makes it particularly useful in tasks that require an understanding of word meanings within sentences.
6. Pre-trained Models:
    - Pre-trained ELMo models are available for various languages and domains, making it easy to leverage these embeddings in NLP projects.
    - While Word2Vec and GloVe models are also pre-trained and widely used, they may not capture context as effectively as ELMo.

# BERT EMBEDDINGS

BERT (Bidirectional Encoder Representations from Transformers) is a powerful pre-trained language model that can be used for various natural language processing tasks, including text classification, named entity recognition, and question-answering.

BERT (Bidirectional Encoder Representations from Transformers) is a revolutionary natural language processing (NLP) model introduced by Google AI in 2018. It has had a profound impact on a wide range of NLP tasks due to its ability to generate contextually rich word embeddings

## **What are BERT Embeddings?**

BERT embeddings are dense vector representations of words, subwords, or entire sentences that capture the contextual information of the input text. Unlike traditional word embeddings like Word2Vec or GloVe, BERT embeddings are contextually aware, meaning they consider the surrounding words when generating representations. BERT can create embeddings for both individual words and entire sentences, which makes it versatile for various NLP tasks.

1. **Pre-training and Fine-tuning:**
    
    BERT is a two-step process:
    
    a. **Pre-training:** BERT is pre-trained on a massive corpus of text. During this phase, it learns to predict missing words in a sentence (masked language modeling) and to understand the relationships between words in a bidirectional manner. This pre-training results in a general-purpose language model that captures a broad range of language knowledge.
    
    b. **Fine-tuning:** BERT can be fine-tuned on specific NLP tasks. Fine-tuning involves training BERT on a smaller, task-specific dataset with a task-specific objective, such as text classification or named entity recognition. Fine-tuning makes BERT adaptable to various downstream NLP tasks.
    
2. **BERT Models and Variants:**
    
    BERT comes in various model sizes and variants. Some popular variants include:
    
    - B**ert-base-uncased**: The base BERT model with 12 layers and 110M parameters, trained on uncased text.
    - B**ert-large-uncased**: A larger version of BERT with 24 layers and 340M parameters.
    - **Multilingual BERT (mBERT)**: Trained in multiple languages.
    - **BioBERT**: Fine-tuned for biomedical text.
    - **ClinicalBERT**: Fine-tuned for clinical text.
    
    The choice of the BERT model depends on the specific task and available computational resources.
    
3. **Tokenization:**
    
    BERT uses WordPiece tokenization, which breaks text into subwords and assigns a unique ID to each subword. This subword tokenization helps BERT handle out-of-vocabulary words and different word forms effectively.
    
4. **Obtaining BERT Embeddings:**
    
    To obtain BERT embeddings for a text sequence, you typically follow these steps:
    
    - Tokenize the input text into subwords.
    - Convert the subwords into token IDs using the BERT model's tokenizer.
    - Pass the token IDs through the BERT model.
    - Extract embeddings from the hidden layers.
5. **Applications of BERT Embeddings:**
    
    BERT embeddings are widely used in various NLP tasks, including but not limited to:
    
    - Text classification
    - Named entity recognition
    - Question-answering
    - Sentiment analysis
    - Machine translation
    - Text summarization
    - Semantic similarity measurement
6. **Contextual Information:**
    
    BERT embeddings capture contextual information by considering the entire input sequence. For instance, the word "bank" in the sentence "I went to the bank" is represented differently from "bank" in "I deposited money in the bank."
    
7. **Limitations and Challenges:**
    
    While BERT is a powerful model, it has some limitations:
    
    - It can be computationally expensive.
    - Fine-tuning requires task-specific labeled data.
    - It may not perform well on languages or domains it was not trained on.

# Sentence Transformers

Sentence Transformers are a specific type of transformer-based model designed for encoding sentences or text snippets into dense vector representations, often referred to as sentence embeddings. These embeddings capture the semantic information and context of the text, making them highly useful for various natural language processing (NLP) tasks, including semantic similarity comparison, semantic search, clustering, and more.

## Why Sentence Transformers?

Sentence Transformers are a valuable advancement in natural language processing (NLP) for several reasons:

1. **Semantic Understanding**: Sentence Transformers excel at capturing the semantic content and context of sentences. They encode sentences into dense vector representations that reflect their underlying meaning, making them highly suitable for understanding and comparing text at a semantic level.
2. **Semantic Similarity**: Sentence Transformers are designed to compute semantic similarity between sentences. This is crucial for various NLP applications, such as information retrieval, question-answering, text summarization, and sentiment analysis. By providing reliable similarity scores, they enable more accurate results in these tasks.
3. **Efficiency**: Once sentence embeddings are computed, they can be stored and efficiently used for semantic similarity calculations. This is especially important in scenarios where large volumes of text need to be compared, as it significantly reduces computation time.
4. **Semantic Search**: Sentence Transformers enable semantic search, allowing users to find information based on the meaning of their queries rather than relying solely on exact keyword matches. This is valuable for search engines, e-commerce platforms, and recommendation systems.
5. **Clustering**: These embeddings can be used for text clustering, helping to group similar pieces of text together. This is beneficial for topic modeling, content organization, and content recommendation systems.
6. **Cross-Modal Tasks**: Sentence Transformers are not limited to text-to-text comparisons. They can be used in cross-modal tasks where text needs to be compared with other data types, such as images, audio, or video, enabling multimodal search and recommendation systems.
7. **Pretrained Models**: Like other transformer-based models, Sentence Transformers benefit from pretraining on large text corpora. This means they have a strong foundation in understanding natural language and are then fine-tuned for specific tasks, which often leads to excellent performance.
8. **User-Friendly Integration**: Libraries and APIs, such as the sentence-transformers library and Hugging Face's Transformers, make it easy for developers and researchers to integrate Sentence Transformers into their NLP projects.
9. **Continuous Improvement**: The field of Sentence Transformers is actively evolving. Newer models and approaches are regularly introduced, pushing the boundaries of what is possible in terms of semantic understanding and similarity computation.

## SBERT

SBERT, or Sentence-BERT, is a specific variant of Sentence Transformers that focuses on producing high-quality sentence embeddings for semantic similarity tasks. It was introduced by Nils Reimers and Iryna Gurevych in their 2019 paper titled "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.”

### Some Key Features and Characteristics of SBERT:

1. **Siamese Network Architecture**: SBERT employs a Siamese network architecture, which consists of two identical BERT models (or other transformer-based models) with shared weights. This architecture allows SBERT to encode pairs of sentences simultaneously and learn to produce embeddings that capture semantic similarity.
2. **Semantic Similarity Learning**: SBERT is trained to produce sentence embeddings in such a way that similar sentences result in embeddings that are close in the vector space, while dissimilar sentences are separated. This training is typically performed on large datasets with sentence pairs and similarity labels.
3. **Pooling Strategies**: SBERT utilizes various pooling strategies to aggregate token-level embeddings into sentence-level embeddings. Common pooling methods include mean pooling (averaging token embeddings), max pooling (selecting the maximum values), and [CLS] token pooling (using the special [CLS] token's embedding).
4. **Efficient Sentence Comparison**: SBERT embeddings can be precomputed and efficiently compared to measure semantic similarity between sentences. This is advantageous in applications like search engines, where rapid similarity assessments are essential.
5. **Multiple Languages**: SBERT models are available for multiple languages, enabling semantic similarity calculations in various linguistic contexts.
6. **Clustering and Search**: SBERT embeddings can be used for clustering similar sentences and for semantic search, where users can search for documents or text based on the meaning of their queries rather than exact keyword matches.
7. **Efficiency and Scalability**: SBERT has demonstrated significant improvements in efficiency and scalability for semantic similarity tasks, particularly in comparison to traditional BERT models.

Sentence Transformers, such as SBERT, are specialized models for encoding sentences into semantic embeddings for tasks like measuring sentence similarity. Unlike BERT, SBERT employs a Siamese architecture, which involves two identical BERT models that share network weights. These models process sentence pairs and produce embeddings that are fine-tuned for semantic similarity. Mean pooling is used to convert token-level embeddings into a single sentence-level embedding. Despite the appearance of two models, it's essentially a single BERT model trained to handle sentence pairs efficiently and extract meaningful sentence representations for semantic comparison.

![2425dc0efd3f73a0bf57b3bf85a091c78619ec2c-1920x1110.png](SPRINT%204%20Text%20Representations%20and%20Vector%20Databases%206567ba269f08494385734d7f61364569/2425dc0efd3f73a0bf57b3bf85a091c78619ec2c-1920x1110.png)

## SBERT MODELS:

1. **bert-base-nli-mean-tokens**: This is one of the earlier SBERT models. It is based on the BERT architecture and is fine-tuned for sentence embeddings. While it may not be the most advanced, it still serves as a good baseline model.
2. **all-mpnet-base-v2**: This model is based on the MPNet architecture, a variant of BERT, and is known for its excellent performance. It offers a larger context window, allowing it to encode longer sequences. The "v2" version indicates improvements over the original model.
3. **all-roberta-large-v1**: This model is based on the RoBERTa architecture, which is an optimized version of BERT. It is a larger model and is well-suited for various NLP tasks, including sentence embeddings.
4. **all-MiniLM-L12-v1**: MiniLM is a smaller variant of BERT designed for efficient processing and reduced resource requirements. This SBERT model is a good choice when you need to balance performance and computational efficiency.
5. **paraphrase-MiniLM-L6-v2**: This model is based on MiniLM but is specifically fine-tuned for paraphrase detection and semantic similarity tasks. It is suitable for tasks where identifying paraphrases is important.