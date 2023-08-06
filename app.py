from flask import Flask, render_template, request
import gunicorn
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

try:
    nlp = spacy.load("en_core_web_sm")
except:  # If not present, we download
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def text_summarizer(raw_text):
    """
    Summarizes a text
    """

    docx = nlp(raw_text)
    keyword = list()
    words = list()
    stop_words = list(STOP_WORDS)

    pos_tag = ["PROPN", "ADJ", "NOUN", "VERB"]

    for token in docx:
        # print(token)
        if token.text not in punctuation:
            words.append(token.text)
        if (token.text in stop_words) or (token.text in punctuation):
            continue
        if token.pos_ in pos_tag:
            keyword.append(token.text)
    # Build Word Frequency
    # word.text is tokenization in spacy
    freq_word = Counter(keyword)

    # Normalize
    max_freq = freq_word.most_common(1)[0][1]

    for word in freq_word.keys():
        freq_word[word] = freq_word[word] / max_freq

    sentence_scores = dict()

    # Sentence Scores
    for sentence in docx.sents:

        for word in sentence:
            # print(word)
            if word.text in freq_word.keys():
                if sentence in sentence_scores.keys():
                    sentence_scores[sentence] += freq_word[word.text]
                else:
                    sentence_scores[sentence] = freq_word[word.text]

    summarized_sentences = nlargest(3, sentence_scores, key=sentence_scores.get)

    summary = [w.text for w in summarized_sentences]
    final_summary = " ".join(summary)
    return len(words), final_summary

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/summarize_text', methods=["POST"])
def summarize_text():
    if request.method == "POST":
        input_text = request.form["input_text"]
        word_count, summarized_text = text_summarizer(input_text)

        return render_template(
            "summary.html",
            input_text=input_text,
            summarized_text=summarized_text,
            word_count=word_count,
        )

    else:
        return render_template(
            "summary.html",
            input_text='',
            summarized_text='',
            word_count='',)

if __name__ == "__main__":
    app.run(debug=True)