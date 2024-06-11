import nltk
from flask import Flask, render_template, request, send_from_directory, make_response, jsonify
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from gtts import gTTS
from nltk.corpus import wordnet
import torch
from sentence_splitter import SentenceSplitter, split_text_into_sentences
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def text_summarizer(raw_text, max_summary_sentences):
    try:
        docx = nlp(raw_text)
        stop_words = set(STOP_WORDS)
        keyword = [token.text.lower() for token in docx if token.text.lower() not in stop_words and token.text not in punctuation]
        # Build Word Frequency
        word_freq = {}
        for word in keyword:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
        # Normalize
        max_freq = max(word_freq.values())
        word_freq = {word: freq / max_freq for word, freq in word_freq.items()}
        # Calculate sentence scores
        sentence_scores = {}
        for sentence in docx.sents:
            sentence_length = sum(1 for token in sentence if token.text not in punctuation)
            sentence_score = sum(word_freq.get(word.text.lower(), 0) for word in sentence)
            sentence_scores[sentence] = sentence_score / sentence_length if sentence_length > 0 else 0
        # Select top sentences for summary
        summarized_sentences = nlargest(max_summary_sentences, sentence_scores, key=sentence_scores.get)
        summarized_text = " ".join(sentence.text for sentence in summarized_sentences)
        word_count = len(raw_text.split())
        final_word_count = len(summarized_text.split())
        return word_count, summarized_text, final_word_count, None
    except Exception as e:
        error_message = f"Error occurred during text summarization: {str(e)}"
        return None, None, None, error_message
app = Flask(__name__, static_folder='static')

@app.route('/paraphrase_form', methods=['POST'])
def paraphrase_form():
    if request.method == 'POST':
        input_text_paraphrase = request.form.get('input_text_paraphrase', '')
        if input_text_paraphrase:
            splitter = SentenceSplitter(language='en')
            sentence_list = splitter.split(input_text_paraphrase)
            paraphrase = []
            for i in sentence_list:
                a = paraphrase_text(i,1)
                paraphrase.append(a)
            paraphrase2 = [' '.join(x) for x in paraphrase]
            paraphrase3 = [' '.join(x for x in paraphrase2) ]
            paraphrased_text = str(paraphrase3).strip('[]').strip("'")
            return render_template('paraphrase_form.html', input_text_paraphrase=input_text_paraphrase, paraphrased_text=paraphrased_text)
        else:
            return render_template('paraphrase_form.html', input_text_paraphrase="", paraphrased_text="Input text is empty. Please provide some text.")
    else:
        return "Invalid request method. Please use POST."


def paraphrase_text(input_text,num_return_sequences=1):
  
    batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
    paraphrased_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return paraphrased_text

@app.route('/', methods=['GET', 'POST'])
def index():
    meanings = {}
    if request.method == 'POST':
        word = request.form.get('word')
        if word:
            meanings[word] = get_word_meaning(word)
    return render_template('index.html', meanings=meanings)


@app.route('/summarize_text', methods=['POST'])
def summarize_text():
    if request.method == "POST":
        input_text = request.form["input_text"]
        if input_text:
            max_summary_sentences = int(request.form.get("max_summary_sentences", 3))
            word_count, summarized_text, final_word_count, error_message = text_summarizer(input_text, max_summary_sentences)
            if error_message:
                return jsonify({"error": error_message})
            # Save the summarized text to a temporary file in the static folder
            temp_filename = "temp_summary.mp3"
            temp_file_path = os.path.join(app.static_folder, temp_filename)
            tts = gTTS(summarized_text, lang='en')
            tts.save(temp_file_path)
            return render_template(
                "summary.html",
                input_text=input_text,
                summarized_text=summarized_text,
                word_count=word_count,
                final_word_count=final_word_count,
                temp_filename=temp_filename,
                max_summary_sentences=max_summary_sentences
            )
        else:
            return render_template("summary.html",input_text="",summarized_text="Input text is empty. Please provide some text.")
    else:
        return render_template(
            "summary.html",
            input_text='',
            summarized_text='',
            word_count='',
            final_word_count=''
        )
@app.route('/download_summary')
def download_summary():
    input_text = request.args.get("input_text")
    max_summary_sentences = int(request.args.get("max_summary_sentences", 3))
    word_count, summarized_text, final_word_count, error_message = text_summarizer(input_text, max_summary_sentences)
    if error_message:
        return jsonify({"error": error_message})
    # Creating a response with the summarized text as a file attachment
    response = make_response(summarized_text)
    response.headers["Content-Disposition"] = "attachment; filename=summarized_text.txt"
    response.headers["Content-Type"] = "text/plain"
    return response

# Route to serve the static audio file
@app.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

def get_word_meaning(word):
    synsets = wordnet.synsets(word)
    meanings = []
    for synset in synsets:
        meanings.extend(synset.definition().split(';'))
    return meanings

if __name__ == "__main__":
    app.run(debug=True)
