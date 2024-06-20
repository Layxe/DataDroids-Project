# Results

These are our results for the the [Contradictory, My Dear Watson](https://www.kaggle.com/competitions/contradictory-my-dear-watson/data) kaggle challenge on the [Contradiction Dataset](https://www.kaggle.com/competitions/contradictory-my-dear-watson/data). All models were evaluated on 20% of the train data with random state since the challenge does not provide ground truth for the test data. We also provide our results [here](\final-project\output). For english training data only, we managed to get over 95% accuracy on our evaluation data. 

## Table Of Contents

- [Results](#results)
    - [Table Of Contents](#table-of-contents)
    - [Probabilistic Approach](#probabilistic-approach)
    - [Forest-based Models](#forest-based-models)
        - [Comparison](#comparison)
        - [Decision Tree](#decision-tree)
        - [Random Forest](#random-forest)
        - [Gradient Boosting](#gradient-boosting)
    - [Deep Learning Methods](#deep-learning-methods)
        - [Results on all languages](#results-on-all-languages)
            - [Comparison](#comparison-1)
            - [Bert all](#bert-all)
            - [Bert finetuned](#bert-finetuned)
            - [Roberta all](#roberta-all)
            - [Roberta finetuned](#roberta-finetuned)
        - [Results on English Data](#results-on-english-data)
            - [Comparison](#comparison-2)
            - [Bert all](#bert-all-1)
            - [Bert finetuned](#bert-finetuned-1)
            - [Roberta all](#roberta-all-1)
            - [Roberta finetuned](#roberta-finetuned-1)
        - [Results on Cross Dataset Trainng](#results-on-cross-dataset-training)

## Probabilistic Approach

In our [data analysis](/final-project/ProjectWork_1.ipynb) we found out, that the dataset consists of 20617 tokens where some tokens appeared more frequently in particular classes. For this reason our first approach was to use the most frequent tokens for each label to identify `contradiction`, `neutral` and `entailment` by their token appearance. We managed to get an Accuracy of 41.44 percent with this approach.

## Forest-based Models

Second, we trained several forest based methods for classification by concatenating the first and second statement, using a separator `<|>`. We used GridSearch with different parameters, which you can find in our [source code](/final-project/ProjectWork_2.ipynb).

### Comparison

| Metric              |Decision Tree | Random Forest   | Gradient Boosting |
|---------------------|--------------|-----------------|-------------------|
|Mean Accuracy        |0.3909        |0.3803           |0.4030
|Standard Deviation   |0.0027        |0.0023           |0.0056
|Best Accuracy        |0.3849        |0.3762           |0.4154
|Best Hyperparameters |{'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 40} | {'criterion': 'gini', 'max_depth': 32, 'min_samples_split': 40, 'n_estimators': 200} | {'max_depth': 8, 'min_samples_split': 20, 'n_estimators': 25} |

![accuracy_forests](/docs/accuracy_forests.png)

### Decision Tree

![decision_tree_confusion](/docs/decision_tree_confusion.png)

### Random Forest

![random_forest_confusion](/docs/random_forest_confusion.png)

### Gradient Boosting

![gradient_boosting_confusion](/docs/decision_tree_confusion.png)

## Deep Learning Methods

We trained several deep learning models including BERT and ROBERTA models.

The code for training can be found in [this notebook](/final-project/ProjectWork_5.ipynb) and evaluation code in our [evaluation notebook](/final-project/EvaluateTransformers.ipynb). We finetuned each models first (marked as `_finetuned`) and then trained the model from ground (marked as `_all`). For each model we also provide confusion matrices and PCA embeddings.

### Results on all languages

First we trained our models on the whole dataset. Here are our results:

#### Comparison

| Metric | BERT_all | BERT_finetuned | ROBERTA_all | ROBERTA_finetuned |
|--------|----------|----------------|-------------|-------------------|
|Accuracy|0.6847    |0.4546          |0.8589       | 0.4576            |
|F1-Score|0.6832    |0.4392          |0.8586       | 0.4570 	       |
|Runtime |12.39 s   |12.33 s         |39.80 s      | 39.18 s           |

#### BERT all

![confusion_matrix](/final-project/output/all_languages/bert-checkpoint-2752/confusion-matrix.png)

![pca-embeddings](/final-project/output/all_languages/bert-checkpoint-2752/pca-embeddings.png)

#### BERT finetuned

![confusion_matrix](/final-project/output/all_languages/bert-checkpoint-374-finetuned/confusion-matrix.png)

![pca-embeddings](/final-project/output/all_languages/bert-checkpoint-374-finetuned/pca-embeddings.png)

#### ROBERTA all
![confusion_matrix](/final-project/output/all_languages/roberta-checkpoint-21816-all/confusion-matrix.png)

![pca-embeddings](/final-project/output/all_languages/roberta-checkpoint-21816-all/pca-embeddings.png)

#### ROBERTA finetuned
![confusion_matrix](/final-project/output/all_languages/roberta-checkpoint-4816-finetuned/confusion-matrix.png)

![pca-embeddings](/final-project/output/all_languages/roberta-checkpoint-4816-finetuned/pca-embeddings.png)

### Results on English Data

Furthermore, we trained all models on the same data, but all languages except english have been removed. Here are our results:

#### Comparison

| Metric | BERT_all | BERT_finetuned | ROBERTA_all | ROBERTA_finetuned |
|--------|----------|----------------|-------------|-------------------|
|Accuracy|0.6252    |0.4701          |0.9512       | 0.4716            |
|F1-Score|0.6254    |0.4651          |0.9509       | 0.4577 	       |
|Runtime |4.63 s    |4.64 s          |8.75 s       | 8.68 s            |

#### BERT all

![confusion_matrix](/final-project/output/only_english/bert-checkpoint-2752/confusion-matrix.png)

![pca-embeddings](/final-project/output/only_english/bert-checkpoint-2752/pca-embeddings.png)

#### BERT finetuned

![confusion_matrix](/final-project/output/only_english/bert-checkpoint-374-finetuned/confusion-matrix.png)

![pca-embeddings](/final-project/output/only_english/bert-checkpoint-374-finetuned/pca-embeddings.png)

#### ROBERTA all
![confusion_matrix](/final-project/output/only_english/roberta-checkpoint-21816-all/confusion-matrix.png)

![pca-embeddings](/final-project/output/only_english/roberta-checkpoint-21816-all/pca-embeddings.png)

#### ROBERTA finetuned
![confusion_matrix](/final-project/output/only_english/roberta-checkpoint-4816-finetuned/confusion-matrix.png)

![pca-embeddings](/final-project/output/only_english/roberta-checkpoint-4816-finetuned/pca-embeddings.png)


### Results on Cross Dataset Training

The next step for us is cross dataset training, combining further datasets. We used a sample of the MNLI dataset giving us 8x more data (~100.000) training data. The test dataset stays the same as in previous examples. Due to the long training time (3h per epoch) we started only trying out ROBERTA.

After training for 5 more epochs we evaluated the accuracy for our origin test data. The hyperparameters were the same as in the previous training.

| Metric   | ROBERTA All Languages | Roberta Only English |
| -------- | --------------------- | -------------------- |
| Accuracy | 0.68783               | 0.86098              |
| F1-Score | 0.68789               | 0.8597               |
| Runtime  | 39.33 s               | 8.72 s               |
