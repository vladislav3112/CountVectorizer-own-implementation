from collections import Counter


class CountVectorizer:
    """
    Класс CountVectorizer, имеющий
    - метод fit_transform
    - метод get_feature_names
    """

    text_corpus = []

    def get_feature_names(self) -> list:
        """
        * ничего не принимает
        * возвращает список фичей (уникальных слов из корпуса)
        """
        corpus = self.text_corpus
        per_text_corpus_features = []
        for text in corpus:
            per_text_corpus_features.append(set(text.lower().split()))
        whole_corpus_features = set().union(*per_text_corpus_features)
        return list(whole_corpus_features)

    def fit_transform(self, corpus: list) -> list:
        """
        * принимает текстовый корпус
        * возвращает терм-документную матрицу
        """
        self.text_corpus = corpus
        features = self.get_feature_names()
        cnt_matrix = []

        for text in corpus:
            lower_text = text.lower()
            all_counts = Counter(lower_text.split())
            cnt = [all_counts[feature] for feature in features]
            cnt_matrix.append(cnt)
        return cnt_matrix
