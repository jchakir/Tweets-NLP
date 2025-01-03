# Tweets and NLP with GridSearch: README

## 🚀 Introduction

This Jupyter notebook is designed to preprocess, analyze, and model tweets using advanced natural language processing (NLP) techniques. It incorporates grid search to optimize parameters for machine learning models, ensuring efficient and accurate analysis.

### Objectives

- Cleaning and preprocessing tweet data.
- Generating vector representations of text.
- Using machine learning models.
- Optimizing model parameters using grid search.

This Notebook is valuable for developers, data scientists, and enthusiasts who want to explore NLP techniques on social media data.

---

## 📋 Requirements

To run this notebook successfully, ensure you have the following:

### Python Libraries

- `nltk`
- `pandas`, `numpy`
- `fasttext`
- `scikit-learn`
- `pyspellchecker`

You can install these libraries manually if needed:

```bash
pip install nltk pandas numpy fasttext scikit-learn pyspellchecker
```

---

## 🛠️ Installation

1. Clone the repository containing the notebook:
   ```bash
   git clone git@github.com:jchakir/Tweets-NLP.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Tweets-NLP
   ```
3. Install the required libraries as specified in the requirements section.
4. Open the notebook in Jupyter:
   ```bash
   jupyter notebook tweets-with-gridsearch.ipynb
   ```

---

## 📖 How to Use

1. **Prepare Your Data**: Format your tweet dataset correctly and place it in the designated directory.
2. **Run the Notebook**: Execute each cell sequentially, from importing libraries to final model evaluation.
3. **Understand the Workflow**:
   - **Preprocessing**: Cleans and prepares the text data.
   - **Feature Engineering**: Converts text into numerical formats.
   - **Model Training**: Applies machine learning models.
   - **Hyperparameter Tuning**: Uses grid search for optimal settings.
4. **Analyze Results**: Review the outputs to refine your approach as needed.

---

## ✨ Key Features

- **Efficient Text Cleaning**: Handles noise and irrelevant components in tweets.
- **Advanced Vectorization**: Employs multiple techniques for text-to-numerical transformation.
- **Automated Optimization**: Grid search ensures model parameters are fine-tuned.
- **Modular Workflow**: Structured and reusable components.

---

## 🧪 Technologies Used

### Preprocessing

- **Text Cleaning**: Removal of special characters, stopwords, and irrelevant data.
- **Tokenization**: Splitting text into meaningful units.
- **Stemming and Lemmatization**: Reducing words to their base forms using `nltk`.

### Vectorization

- **Bag-of-Words (BoW)**: Converts text into numerical format based on word frequency.
- **TF-IDF**: Weights words based on their importance in the document.
- **FastText Word Embeddings**: Generates dense vector representations using FastText.

### Machine Learning

- **Logistic Regression**: For binary classification.
- **Random Forest**: A robust ensemble learning method.
- **Support Vector Machines (SVM)**: Excels in high-dimensional spaces.

### Optimization

- **Grid Search**: Systematic parameter tuning.
- **Cross-Validation**: Ensures robust model evaluation.

---

## 💡 Tips for Effective Use

- Ensure your dataset is sufficiently large and diverse.
- Leverage GPU acceleration where possible to speed up computation.
- Regularly update libraries to access the latest features.

---

# Final Accuracy
```
stemming                    and   binary        vectorizing using   LogisticRegression        model:   0.9272300
stemming+misspelling        and   binary        vectorizing using   LogisticRegression        model:   0.9272300
stemming                    and   word_counts   vectorizing using   LogisticRegression        model:   0.9225352
just_tokenization           and   word_counts   vectorizing using   LogisticRegression        model:   0.9154930
just_tokenization           and   word_counts   vectorizing using   DecisionTreeClassifier    model:   0.9154930
stemming                    and   binary        vectorizing using   BernoulliNB               model:   0.9154930
stemming                    and   word_counts   vectorizing using   BernoulliNB               model:   0.9154930
stemming                    and   tfidf         vectorizing using   BernoulliNB               model:   0.9154930
lemmatization               and   binary        vectorizing using   LogisticRegression        model:   0.9154930
stemming+misspelling        and   word_counts   vectorizing using   LogisticRegression        model:   0.9154930
lemmatization+misspelling   and   binary        vectorizing using   LogisticRegression        model:   0.9154930
lemmatization+stopwords     and   binary        vectorizing using   LogisticRegression        model:   0.9154930
just_tokenization           and   binary        vectorizing using   LogisticRegression        model:   0.9131455
stemming+misspelling        and   binary        vectorizing using   BernoulliNB               model:   0.9131455
stemming+misspelling        and   word_counts   vectorizing using   BernoulliNB               model:   0.9131455
stemming+misspelling        and   tfidf         vectorizing using   BernoulliNB               model:   0.9131455
lemmatization               and   word_counts   vectorizing using   LogisticRegression        model:   0.9107981
lemmatization+misspelling   and   word_counts   vectorizing using   LogisticRegression        model:   0.9084507
lemmatization+stopwords     and   word_counts   vectorizing using   LogisticRegression        model:   0.9084507
stemming                    and   tfidf         vectorizing using   LogisticRegression        model:   0.9061033
lemmatization+misspelling   and   binary        vectorizing using   BernoulliNB               model:   0.9061033
lemmatization+misspelling   and   word_counts   vectorizing using   BernoulliNB               model:   0.9061033
lemmatization+misspelling   and   tfidf         vectorizing using   BernoulliNB               model:   0.9061033
lemmatization+stopwords     and   tfidf         vectorizing using   DecisionTreeClassifier    model:   0.9061033
just_tokenization           and   binary        vectorizing using   DecisionTreeClassifier    model:   0.9037559
just_tokenization           and   tfidf         vectorizing using   DecisionTreeClassifier    model:   0.9037559
stemming                    and   tfidf         vectorizing using   DecisionTreeClassifier    model:   0.9037559
lemmatization               and   word_counts   vectorizing using   DecisionTreeClassifier    model:   0.9037559
lemmatization+misspelling   and   tfidf         vectorizing using   DecisionTreeClassifier    model:   0.9037559
lemmatization+stopwords     and   binary        vectorizing using   BernoulliNB               model:   0.9037559
lemmatization+stopwords     and   word_counts   vectorizing using   BernoulliNB               model:   0.9037559
lemmatization+stopwords     and   tfidf         vectorizing using   BernoulliNB               model:   0.9037559
lemmatization               and   binary        vectorizing using   DecisionTreeClassifier    model:   0.9014085
stemming+misspelling        and   tfidf         vectorizing using   LogisticRegression        model:   0.9014085
just_tokenization           and   binary        vectorizing using   BernoulliNB               model:   0.8990610
just_tokenization           and   word_counts   vectorizing using   BernoulliNB               model:   0.8990610
just_tokenization           and   tfidf         vectorizing using   LogisticRegression        model:   0.8990610
just_tokenization           and   tfidf         vectorizing using   BernoulliNB               model:   0.8990610
lemmatization               and   binary        vectorizing using   BernoulliNB               model:   0.8990610
lemmatization               and   word_counts   vectorizing using   BernoulliNB               model:   0.8990610
lemmatization               and   tfidf         vectorizing using   BernoulliNB               model:   0.8990610
lemmatization+misspelling   and   word_counts   vectorizing using   DecisionTreeClassifier    model:   0.8990610
lemmatization+misspelling   and   tfidf         vectorizing using   LogisticRegression        model:   0.8990610
lemmatization+misspelling   and   binary        vectorizing using   DecisionTreeClassifier    model:   0.8967136
stemming                    and   binary        vectorizing using   DecisionTreeClassifier    model:   0.8943662
lemmatization               and   tfidf         vectorizing using   LogisticRegression        model:   0.8943662
stemming+misspelling        and   tfidf         vectorizing using   DecisionTreeClassifier    model:   0.8943662
stemming                    and   word_counts   vectorizing using   DecisionTreeClassifier    model:   0.8920188
lemmatization               and   tfidf         vectorizing using   DecisionTreeClassifier    model:   0.8920188
stemming+misspelling        and   binary        vectorizing using   DecisionTreeClassifier    model:   0.8896714
stemming+misspelling        and   word_counts   vectorizing using   DecisionTreeClassifier    model:   0.8896714
lemmatization+stopwords     and   tfidf         vectorizing using   LogisticRegression        model:   0.8849765
lemmatization+stopwords     and   word_counts   vectorizing using   DecisionTreeClassifier    model:   0.8779343
lemmatization+stopwords     and   binary        vectorizing using   DecisionTreeClassifier    model:   0.8708920
stemming                    and   word2vec      vectorizing using   LogisticRegression        model:   0.8356808
stemming+misspelling        and   word2vec      vectorizing using   LogisticRegression        model:   0.8333333
just_tokenization           and   word2vec      vectorizing using   LogisticRegression        model:   0.8262911
lemmatization+misspelling   and   word2vec      vectorizing using   LogisticRegression        model:   0.8239437
lemmatization               and   word2vec      vectorizing using   LogisticRegression        model:   0.8192488
lemmatization+stopwords     and   word2vec      vectorizing using   LogisticRegression        model:   0.8098592
stemming                    and   word2vec      vectorizing using   DecisionTreeClassifier    model:   0.8028169
lemmatization               and   word2vec      vectorizing using   DecisionTreeClassifier    model:   0.7910798
stemming                    and   word_counts   vectorizing using   GaussianNB                model:   0.7887324
just_tokenization           and   word_counts   vectorizing using   GaussianNB                model:   0.7863850
stemming                    and   binary        vectorizing using   GaussianNB                model:   0.7840376
lemmatization               and   word_counts   vectorizing using   GaussianNB                model:   0.7816901
just_tokenization           and   binary        vectorizing using   GaussianNB                model:   0.7793427
lemmatization               and   binary        vectorizing using   GaussianNB                model:   0.7793427
stemming+misspelling        and   word2vec      vectorizing using   DecisionTreeClassifier    model:   0.7793427
lemmatization+misspelling   and   word2vec      vectorizing using   DecisionTreeClassifier    model:   0.7793427
lemmatization+stopwords     and   word_counts   vectorizing using   GaussianNB                model:   0.7769953
lemmatization+misspelling   and   word_counts   vectorizing using   GaussianNB                model:   0.7746479
lemmatization+stopwords     and   binary        vectorizing using   GaussianNB                model:   0.7746479
stemming+misspelling        and   word_counts   vectorizing using   GaussianNB                model:   0.7699531
lemmatization+misspelling   and   binary        vectorizing using   GaussianNB                model:   0.7699531
just_tokenization           and   word2vec      vectorizing using   DecisionTreeClassifier    model:   0.7676056
just_tokenization           and   word2vec      vectorizing using   BernoulliNB               model:   0.7676056
lemmatization+misspelling   and   word2vec      vectorizing using   BernoulliNB               model:   0.7676056
stemming+misspelling        and   binary        vectorizing using   GaussianNB                model:   0.7652582
stemming                    and   tfidf         vectorizing using   GaussianNB                model:   0.7558685
just_tokenization           and   tfidf         vectorizing using   GaussianNB                model:   0.7511737
stemming                    and   word2vec      vectorizing using   BernoulliNB               model:   0.7511737
lemmatization+stopwords     and   tfidf         vectorizing using   GaussianNB                model:   0.7511737
lemmatization               and   tfidf         vectorizing using   GaussianNB                model:   0.7488263
lemmatization+misspelling   and   tfidf         vectorizing using   GaussianNB                model:   0.7464789
lemmatization               and   word2vec      vectorizing using   BernoulliNB               model:   0.7417840
stemming+misspelling        and   tfidf         vectorizing using   GaussianNB                model:   0.7323944
lemmatization+stopwords     and   word2vec      vectorizing using   DecisionTreeClassifier    model:   0.7230047
stemming+misspelling        and   word2vec      vectorizing using   BernoulliNB               model:   0.6924883
lemmatization+stopwords     and   word2vec      vectorizing using   BernoulliNB               model:   0.6455399
lemmatization               and   word2vec      vectorizing using   GaussianNB                model:   0.6079812
lemmatization+misspelling   and   word2vec      vectorizing using   GaussianNB                model:   0.6079812
lemmatization+stopwords     and   word2vec      vectorizing using   GaussianNB                model:   0.6056338
stemming+misspelling        and   word2vec      vectorizing using   GaussianNB                model:   0.6032864
just_tokenization           and   word2vec      vectorizing using   GaussianNB                model:   0.5938967
stemming                    and   word2vec      vectorizing using   GaussianNB                model:   0.5868545
```
