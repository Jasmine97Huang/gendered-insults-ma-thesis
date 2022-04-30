import pandas as pd
from ast import literal_eval
import numpy as np
import spacy
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_LEN = 128
#BERT_PATH = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def convert_gender(g):
    if g not in ['male', 'female', 'group']:
        return 'Non-binary'
    else:
        return g
def gender_encoder(g):
    if g not in ['male', 'female']:
        return 2 #non-binary
    elif g == 'female':
        return 1
    else:
        return 0 #male

def segment_lyrics(file_path):
    '''
    Split a chunk of lyrics into sentences.
    
    Input:
    file_path: path to csv file

    Output:
    None: write file to output directory
    '''
    data = pd.read_csv(file_path, index_col = 0)
    data = data[~data.gender_matched.isin(["Unknown", "group","unknown"])].reset_index(drop = True)
    data['gender_matched'] = data.gender_matched.apply(lambda x: gender_encoder(x))
    data['segmented'] = None

    nlp = spacy.load('en')

    lyrics = data.lyrics.values.tolist()
    for i, para in enumerate(lyrics):
        tokens = nlp(para)
        lst = []
        for sent in tokens.sents:
            lst.append(sent.string.strip())
        #print(lst)
        data.at[i, 'segmented'] = lst
    data['song_index'] = data.index.tolist()
    data.to_csv("output/segmented.csv", index = None)

def tokenize(file_path = "output/segmented.csv"):
    '''
    Input:
    file_path

    Output:
    return pandas dataframe with tokenized_text column added
    '''
    df = pd.read_csv(file_path, index_col = 0)
    segmented = [literal_eval(x) for x in df.segmented.to_list()]
    df['segmented'] = segmented
    df = df.explode('segmented')

    # Create sentence and label lists
    sentences = df.segmented.values.tolist()

    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    #labels = df.gender_matched.values

    #MAX_LEN = 128
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    df['tokenized_text'] = tokenized_texts
    df = df.reset_index(drop = True)
    return df


def set_up_mlm(df, bad_words_path = 'data/bad-words.txt'):
    '''
    Set up Mask Language Model by tokenize corpus, mask bad words and randomly mask 15% other tokens
    Ouput npy file to be used for model training

    Input:
    df: pandas dataframe containing segmented corpus
    bad_words_path: path to bad-words list

    Output:
    None: write file to output directory
    '''
    sentences = df.tokenized_text.to_list()
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in sentences]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    #true label for mlm
    labels = input_ids.copy()

    with open(bad_words_path, 'r') as fp:
        bad_wrds = fp.read().split('\n')
    bad_wrds_ids= [tokenizer.convert_tokens_to_ids(x) for x in bad_wrds]
    bad_wrds_ids = set(bad_wrds_ids)
    mask_arr = np.asarray([wrd in bad_wrds_ids for sent in input_ids for wrd in sent]).reshape(input_ids.shape)
    print("Percentage of tokens that are bad words:",\
          sum([wrd in bad_wrds_ids for sent in input_ids for wrd in sent])/sum([len(sent) for sent in sentences]))
    rand = np.random.rand(input_ids.shape[0], input_ids.shape[1])
    sep_tokens = np.asarray([wrd not in [101, 102, 0] for sent in input_ids for wrd in sent]).reshape(input_ids.shape) #do not mask [CLS], [SEP]
    mask_arr = np.logical_or(mask_arr, np.logical_and((rand < 0.15), sep_tokens))
    print("Number of masked tokens after random masking:", np.sum(mask_arr))
    print("Perc of masked token/total tokens", np.sum(mask_arr)/sum([len(sent) for sent in sentences]))

    selection = []

    for i in range(mask_arr.shape[0]):
        selection.append(np.flatnonzero(mask_arr[i]).tolist())

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [i == 103 for i in seq]
        attention_masks.append(seq_mask)

    attention_masks= np.array(attention_masks, dtype = "int")

    inputs = {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_masks}
    with open('output/inputs.npy', 'wb') as f:
        np.save(f, input_ids)
        np.save(f, labels)
        np.save(f, attention_masks)



