class CountVectorizer:
    text_corpus = []

    def get_feature_names(cls) -> list:
        corpus = cls.text_corpus
        lowercase_sent_corpus = [set(sent.lower().split()) for sent in corpus]
        lowercase_sent_corpus = set().union(*lowercase_sent_corpus)
        return list(lowercase_sent_corpus)

    def fit_transform(cls, corpus: list) -> list:
        cls.text_corpus = corpus
        features = cls.get_feature_names()
        cnt_matrix = []
        for sent in corpus:
            lower_sent = sent.lower()
            cnt = [lower_sent.count(feature) for feature in features]
            cnt_matrix.append(cnt)
        return cnt_matrix