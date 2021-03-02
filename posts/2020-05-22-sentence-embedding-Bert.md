---
layout: post
title: A complete guide to transfer learning from English to other Languages using Sentence Embeddings BERT Models
subtitle: This story aims to provide details on Sentence Embeddings BERT Models to machine learning practitioners
tags: [Machine learning, NLP, Transfer leanrning]
cover-img: /assets/img/transfer.jpg
share-img: /assets/img/transfer.jpg
comments: true
---

# The idea of sentence embedding using Siamese BERT-Networks 📋

**Word Vector** using **Neural vector representations** which has become ubiquitous in all sub fields of Natural Language Processing (NLP) is obviously a familiar field with a lot of people [1]. Among those techniques, **Sentence embedding** idea holds various application potential, which attempts to encode a sentence or short text paragraphs into a fixed length vector (dense vector space) and then the vector is used to evaluate how well their cosine similarities mirror human judgments of semantic relatedness [2].

**Sentence embedding** has wide applications in NLP such as information retrieval, clustering, automatic essay scoring, and for semantic textual similarity. Up to now, there are some popular methodologies for generating fixed length sentence embedding: **Continuous Bag of Words** (CBOW) **Sequence-to-sequence models** (seq2seq) and **Bidirectional Encoder Representations from Transformers** (BERT).

* **CBOW** [3] is an extremely naive approach in which one simple sums (or averages) of the word embeddings encoded by word2vec algorithm to generate the sentence embedding. To be honest, this is such a bad way to capture information in a sentence since it completely disregards the ordering of words.

* **Seq2seq models** [4] like RNNs, LSTMs, GRUs, etc.. on the other hand, take a sequence of items (words, letters, time series, etc) and outputs another sequence of items by processing word embeddings one by one while maintaining a hidden state that stores context information. The embedding produced by encoder captures the context of the input sequence in the form of a hidden state vector and sends it to the decoder, which then produces the output sequence which performs some other task. This approach is capable of differentiating based on word ordering in sentence and thus can potentially provide much richer embeddings than CBOW.

* **BERT model** [5] accomplishes state-of-the-art performance on various sentence classification, sentence-pair regression as well as Semantic Textual Similarity tasks. BERT uses cross-encoder networks that take 2 sentences as input to the transformer network and then predict a target value. However, in a corpus consisting of a large sentences, finding similar pairs of sentences requires a huge inference computations and a lot of time (even with GPU). The pair sentences inference is fairly computationally expensive and scales as O(n.(n-1)/2). For instance, with 1000 sentences we need 1000.(1000 – 1)/2 = 499500 inference computations. Specifically, if we want to filter the most similar questions among 40 million questions on Quora using pair-wise comparison technique with BERT, the query would require over 50 hours to finish running. The figure below explains how the Sentence Pair Classification with BERT works.

To overcome this issue, we can borrow the idea from Computer vision researcher, which uses **Siamese** and **Triplet network** structures to derive a fixed-sized sentence embedding vector and then using a similarity measure like cosine similarity or Manhatten / Euclidean distance to compute semantically similar sentences [6]. This solution was propose by Nils Reimers and Iryna Gurevych from Ubiquitous Knowledge Processing Lab (UKP-TUDA), it called **Sentence-BERT** (SBERT). By using optimized index structures, the running time required for the model to solve the above Quora example can be reduced from 50 hours to a few milliseconds !!!

The architecture of SBERT is simple enough to state. First, the original pair of sentences pass through BERT / RoBERTa to embed fixed sized sentences. Then in the pooling layer, *mean aggregation*, which was proved to have best performance compared to *max* or *CLS* aggregation, is used to generate u and v.

**Classification Objective Function** <br/>
We then concatenate the embeddings as follows: (u, v, ‖u-v‖), multiply by a trainable weight matrix W∈ℝ³ᴺ ˣ ᴷ, where N is the sentence embedding dimension, and K is the number of labels. We optimize cross-entropy loss.  <br/>
**Softmax(Wt(u, v, |u − v|))**

**Regression Objective Function**  <br/>
n case of regression task, we compute sentence embeddings and the cosine similarity of the respective pairs of sentences. We use mean squared-error loss as the objective function. <br/>
**MSE(Wt(u,v,cosine - sim(u,v))**

**Triplet Objective Function** <br/>
Given an anchor sentence A, a positive sentence P, and a negative sentence N, mathematically. We minimize the loss function of A·P distance metric, and A·N distance metric. Margin ɛ ensures that P is at least ɛ closer to a than N.

# Transfer learning sentence embbeding 📊

So far, we can see that SBERT can be used for information retrieval, clustering, automatic essay scoring, and for semantic textual similarity with incredible time and high accuracy. However, the limitation of SBERT is that it only supports English at the moment while leave blank for other languages. To solve that, we can use the model architecture similar with **Siamese** and **Triplet network** structures to extend SBERT to new language [7].

The idea is simple, first we produces sentence embeddings in English sentence by SBERT, we call **Teacher model**. Then we create new model for our desired language, we call Student model, and this model tries to mimic the **Teacher model**. In other word, the original English sentence will be trained in **Student model** in order to get the vector same as one in **Teacher model**.

As the example below, both “Hello World” and “Hallo Welt” were put through **Student model**, and the model tries to generate two vectors that are similar with the one from **Teacher model**. After training, the **Student model** are expected to have ability for encoding the sentence in both language English and the desired language.

Let’s try from scratch with an example for transfer SBERT English to Japanese.

First of all, we need to install SBERT and MeCab package (the important package to parse the Japanese sentence to meaning word).

{% highlight python linenos %}
!pip install -U sentence-transformers
!pip install mecab-python3
{% endhighlight %}

I used XLM-RoBERTa to create word embedding as **Student model** (of course you can try other BERT pre-train model if you want i.e mBERT), “bert-base-nli-stsb-mean-tokens” from **SentenceTransformer** as **Teacher model** and *mean aggregation* as pooling layer. Other parameter are max_seq_length = 128 and train_batch_size = 64 (if it’s over your RAM limitation, you can reduce batch_size to 32 or 16).

{% highlight python linenos %}
from sentence_transformers import SentenceTransformer, models

###### CREATE MODEL ######
max_seq_length = 128
train_batch_size = 64

# Load teacher model
print("Load teacher model")
teacher_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

# Create student model
print("Create student model")
word_embedding_model = models.Transformer("xlm-roberta-base")

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
{% endhighlight %}

After creating **Teacher model** and **Student model**, we can start to load train, dev, test dataset and training model. The train and test sets is Translated Dataset meanwhile the dev sets is Semantic Text Similarity Dataset following the structure of Transfer learning SBERT architecture. In this example, I will train the model in 20 epochs with learning rate = 2e-5 and epsilon = 1e-6, you can freely to try another hyper parameter to get the optimum results in your language. I also save the model for downstream application, if you only want to play around with this, you can turn it off by set save_bet_model = False.

{% highlight python linenos %}
from sentence_transformers.datasets import ParallelSentencesDataset
from torch.utils.data import DataLoader
from sentence_transformers import SentencesDataset, losses, evaluation, readers


###### Load train sets ######

train_reader = ParallelSentencesDataset(student_model=model, teacher_model=teacher_model)
train_reader.load_data('dataset/train.txt')
train_dataloader = DataLoader(train_reader, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=model)


###### Load dev sets ######

evaluators = []
sts_reader = readers.STSDataReader('dataset/', s1_col_idx=0, s2_col_idx=1, score_col_idx=2)
dev_data = SentencesDataset(examples=sts_reader.get_examples('dev.txt'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
evaluator_sts = evaluation.EmbeddingSimilarityEvaluator(dev_dataloader, name='dev')
evaluators.append(evaluator_sts)


###### Load test sets ######

test_reader = ParallelSentencesDataset(student_model=model, teacher_model=teacher_model)
test_reader.load_data('dataset/test.txt')
test_dataloader = DataLoader(test_reader, shuffle=False, batch_size=train_batch_size)
test_mse = evaluation.MSEEvaluator(test_dataloader, name='test')
evaluators.append(test_mse)


###### Train model ######

output_path = "output/model-" + datetime.now().strftime("%Y-%m-%d")
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1]),
          epochs=20,
          evaluation_steps=1000,
          warmup_steps=10000,
          scheduler='warmupconstant',
          output_path=output_path,
          save_best_model=True,
          optimizer_params= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}
          )
{% endhighlight %}

Finally, let’s enjoy the results. We will evaluate the **Student model** in both English and Japanese corpus with the same meaning of sentences.

{% highlight python linenos %}
import scipy.spatial

# Corpus with example sentences
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ]
corpus_embeddings = model.encode(corpus)

# Query sentences:
queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.']
query_embeddings = model.encode(queries)

# Find the closest 3 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 3
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n======================\n")
    print("Query:", query)
    print("\nTop 3 most similar sentences in corpus:\n")

    for idx, distance in results[0:closest_n]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))
{% endhighlight %}

Let check the ability of Student model in Japanese corpus

{% highlight python linenos %}
import scipy.spatial
import MeCab

# Corpus with example sentences
corpus = ['男は食べ物 を 食べています。',
          '男はパンを食べています。',
          '女の子は赤ん坊を運んでいる。',
          '男が馬に乗っています。',
          '女性がバイオリンを弾いています。',
          '2人の男性が森の中をカートを押しました。',
          '男は囲まれた地面で白い馬に乗っています。',
          '猿が太鼓を弾いています。',
          'チーターが獲物の後ろを走っています。'
          ]
m = MeCab.Tagger("-Owakati")
corpus =  [m.parse(x).strip('\n') for x in corpus]       
corpus_embeddings = model.encode(corpus)

# Query sentences:
queries = ['男はパスタを食べています。', 'ゴリラの衣装を着た人がドラムセットを演奏しています。']
queries =  [m.parse(x).strip('\n') for x in queries]      
query_embeddings = model.encode(queries)

# Find the closest 3 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 3
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n======================\n")
    print("Query:", query)
    print("\nTop 3 most similar sentences in corpus:\n")

    for idx, distance in results[0:closest_n]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))
{% endhighlight %}

# Final Thoughts 📕
We can see that, after distillation knowledge from English SBERT, now our model have the capability to embed new language sentences from any downstream task of NLP. You can try to train SBERT with you own language and let me know the results. Furthermore, the Student model capability is not limited for only 2 language, we can extend the number of languages. Preparing new language dataset and you are ready to go. You can use previous explained code (and now Student model will become Teacher model to teach new Student 😅) . However, as a thumb rule, the accuracy will be reduce a little bit as the trade off for competence.

You can check more detail in [here](https://towardsdatascience.com/a-complete-guide-to-transfer-learning-from-english-to-other-languages-using-sentence-embeddings-8c427f8804a9)


# References
[1] Jeffrey Pennington, Richard Socher, and Christopher Manning, GloVe: Global vectors for word representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP 2014), pages 1532–1543, 2014. <br/>
[2] Quoc Le and Tomas Mikolov, Distributed representations of sentences and documents. In Proceedings of ICML 2014. PMLR, 2014. <br/>
[3] Tomas Mikolov, Kai Chen, Greg Corrado and Jeffrey Dean, Efficient Estimation of Word Representations in Vector Space, 2013. <br/>
[4] Ilya Sutskever, Oriol Vinyals and Quoc V. Le, Sequence to Sequence Learning with Neural Networks, 2014. <br/>
[5] Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, 2019. <br\>
[6] Nils Reimers and Iryna Gurevych, Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, 2019. <br/>
[7] Nils Reimers and Iryna Gurevych, Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation, 2020. <br/>

