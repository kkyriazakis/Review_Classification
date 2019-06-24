import numpy as np
import tensorflow as tf
import _pickle as pickle
import re
from nltk.corpus import stopwords

contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}


def clean_text(sent, remove_stopwords=True):
    sent = sent.lower()  # lowercase sentence
    
    # replace contractions
    sent = sent.split()
    temp = []
    for w in sent:
        if w in contractions:
            temp.append(contractions[w])
        else:
            temp.append(w)
    sent = " ".join(temp)
    
    # remove unwanted characters contractions
    sent = remove_unwanted_char(sent)
	
    # Remove stop words from sentence only
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        sent = sent.split()
        sent = [w for w in sent if not w in stops]
        sent = " ".join(sent)

    return sent


def remove_unwanted_char(txt):
    txt = re.sub(r'[\"\'<>+=_@#%$!&*}{~`/|()^.,]', '', txt)
    txt = txt.replace('-', '')  # removes -
    txt = txt.replace('\\', '')  # removes \
    txt = " ".join(txt.split())  # removes multiple spaces
    return txt


def read(file):
    with open(file, 'rb') as fp:
        return pickle.load(fp)


def text_to_seq(text):
    """Prepare the text for the model"""
    vocab_to_int = read('vocab_to_int')
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


clean_texts = read('Cleaned_text')
vocab_to_int = read('vocab_to_int')
int_to_vocab = read('int_to_vocab')
batch_size = 64

# random = np.random.randint(0, len(clean_texts))
# input_sentence = clean_texts[random]

# Or Enter Your own sentence Below ---------
sentence = "This started out as a 5 star review. The Gear S3 paired with my Galaxy S6 flawlessly and worked well. Calls were clear, notifications from texts, Google Hangouts, etc worked great until November 2017 when an update took out notifications appearing on the watch screen. Incoming call and calendar notifications appear....but no texts. $300+ for a wearable that works in part is not of value. And there has been no effort from a Samsung to resolve the issue. Very disappointing."
input_sentence = clean_text(sentence, remove_stopwords=True)

text = text_to_seq(input_sentence)
checkpoint = "./best_model.ckpt"
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    text_length = loaded_graph.get_tensor_by_name('text_length:0')
    summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    # Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                      summary_length: [np.random.randint(5, 8)],
                                      text_length: [len(text)] * batch_size,
                                      keep_prob: 1.0})[0]

# Remove the padding from the tweet
pad = vocab_to_int["<PAD>"]

print('Original Text:', input_sentence)

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([int_to_vocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))
