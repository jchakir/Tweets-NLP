# Tweets-NLP
1337 Coding School Machine Learning Project for Natural Language Processing and Sentiment Analysis on tweet data.

# Accuracy
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
