from src.features.vectorize import TextVectorizer

def test_vectorizer_save_load(tmp_path):
    vec = TextVectorizer(max_features=100)
    X = vec.fit_transform(["hello world", "hello ai"])
    path = tmp_path / "vec.joblib"
    vec.save(path)

    vec2 = TextVectorizer.load(path)
    X2 = vec2.transform(["hello world"])
    assert X2.shape[1] == X.shape[1]
