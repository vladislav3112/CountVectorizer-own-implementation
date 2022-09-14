from CountVectorizer import CountVectorizer

corpus = [
    "Crock Pot Pasta Never boil pasta again",
    "Pasta Pomodoro Fresh ingredients Parmesan to taste",
]
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(count_matrix)


corpus2 = ["Crock Pot", "Parmesan to taste pot"]
vectorizer2 = CountVectorizer()
count_matrix2 = vectorizer2.fit_transform(corpus2)
print(vectorizer2.get_feature_names())
print(count_matrix2)