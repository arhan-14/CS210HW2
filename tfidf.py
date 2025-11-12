import re
import math
import os

def preprocess_documents():
    with open("tfidf_docs.txt", "r") as file:
        doc_filenames = [line.strip() for line in file if line.strip()]

    with open("stopwords.txt", "r") as stopwords:
        stopword_list = set(word.strip().lower() for word in stopwords if word.strip())

    for filename in doc_filenames:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                text = f.read()

            text = re.sub(r'https?://\S+', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = text.lower()
            text = re.sub(r'\s+', ' ', text).strip()

            words = [w for w in text.split() if w not in stopword_list]

            processed_words = []
            for w in words:
                if w.endswith("ing") and len(w) > 4:
                    w = w[:-3]
                elif w.endswith("ly") and len(w) > 3:
                    w = w[:-2]
                elif w.endswith("ment") and len(w) > 5:
                    w = w[:-4]
                processed_words.append(w)

            processed_text = " ".join(processed_words)

            with open(f"preproc_{filename}", "w", encoding="utf-8") as out_f:
                out_f.write(processed_text)

        except FileNotFoundError:
            print(f"{filename} not found. Skipping.")
    return doc_filenames

def compute_tfidf(doc_filenames):
    preprocessed_docs = {}
    for filename in doc_filenames:
        preproc_filename = f"preproc_{filename}"
        if not os.path.exists(preproc_filename):
            continue
        with open(preproc_filename, "r", encoding="utf-8") as f:
            words = f.read().split()
            preprocessed_docs[filename] = words

    doc_count = len(preprocessed_docs)
    word_in_docs = {}

    for words in preprocessed_docs.values():
        for w in set(words):
            word_in_docs[w] = word_in_docs.get(w, 0) + 1

    for filename, words in preprocessed_docs.items():
        total_words = len(words)
        tf = {w: words.count(w) / total_words for w in set(words)}
        tfidf_scores = {}

        for w, tf_val in tf.items():
            df = word_in_docs[w]
            idf = math.log(doc_count / df) + 1
            tfidf_scores[w] = round(tf_val * idf, 2)

        sorted_scores = sorted(tfidf_scores.items(), key=lambda x: (-x[1], x[0]))
        top5 = sorted_scores[:5]

        with open(f"tfidf_{filename}", "w", encoding="utf-8") as out:
            out.write(str(top5) + "\n")

if __name__ == "__main__":
    doc_files = preprocess_documents()
    compute_tfidf(doc_files)
