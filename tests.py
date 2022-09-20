"""Тесты для CountVectorizer"""
from CountVectorizer import CountVectorizer

if __name__ == "__main__":
    corpus = [
        "Crock Pot Pasta Never boil pasta again",
        "Pasta Pomodoro Fresh ingredients Parmesan to taste",
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    assert set(vectorizer.get_feature_names()) == {
        "parmesan",
        "ingredients",
        "pomodoro",
        "pasta",
        "to",
        "fresh",
        "never",
        "crock",
        "again",
        "boil",
        "taste",
        "pot",
    }
    print(count_matrix)

    corpus2 = ["Crock Pot", "Parmesan to taste pot"]
    vectorizer2 = CountVectorizer()
    count_matrix2 = vectorizer2.fit_transform(corpus2)
    assert set(vectorizer2.get_feature_names()) == {
        "parmesan",
        "crock",
        "to",
        "taste",
        "pot",
    }
    print(count_matrix2)