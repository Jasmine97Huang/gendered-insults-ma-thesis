import pandas as pd
import numpy as np  
import torch
import torch # pip install torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import *
import bert_utils
import preprocessing

import sys
import warnings
warnings.filterwarnings("ignore")
# figure size in inches
rcParams['figure.figsize'] = 11.7,8.27

def temporal_viz(bad_words_file, word, slur, model):
    with open(bad_words_file, 'r') as fp:
        bad_wrds = fp.read().split('\n')
    slur_lst = [w for w in bad_wrds if word in w]
    slur = pd.read_csv(slur)

    fig, axs = plt.subplots(5, 1, figsize=(15, 60),sharex=True, sharey=True)

    for idx, y_b in enumerate(['1970s', '1980s', '1990s', '2000s', '2010s']):
        y_df = slur[slur.year_bin == y_b].reset_index(drop = True)
        #alpha_ = 1
        if len(y_df) > 300:
            print(f"Year {y_b} has too many sentences, will truncate/randomly sample a subset")
            min_num = min(y_df.gender_matched.value_counts().reset_index().gender_matched)
            y_df = y_df.groupby("gender_matched").sample(min(min_num, 100), random_state=7).reset_index(drop=True)
            #alpha_ = 0.3
        print(f"Working on year {y_b}...")
        sents = y_df.segmented.values.tolist()
        print(f"Found {len(sents)} sentences containing the insult...")
        print(f"Fetching embeddings from {model} model")
        emb_lst = []
        na_lst = []
        for i, sent in enumerate(sents):
            emb = bert_utils.main(sent, slur_lst, layers = [-2], model = model) #use the second to last layer as word embedding
            if torch.isnan(emb).any().numpy():
                na_lst.append(i)
            else:
                emb_lst.append(emb)
        print(f"Successfully retrieved {len(emb_lst)} embeddings...")
        embs = torch.stack(emb_lst).numpy()
        embs = embs.astype(np.float64)

        pca= decomposition.PCA(n_components = min(embs.shape[0],44)).fit(embs)
        reduced = pca.transform(embs)
        tsne = manifold.TSNE(n_components = 2).fit_transform(reduced)
        gender = y_df.loc[~y_df.index.isin(na_lst)].gender_matched

        sns.set(font_scale=2)
        sns.scatterplot(tsne[:, 0], tsne[:, 1], hue = gender,\
                                palette="bright", s=40, ax = axs[idx]).set(title=f"Word Embedding Projections ({word}); Year: {y_b}")

    plt.savefig("output/clusterEvolution.png")
    plt.show()

if __name__ == "__main__":
    bad_words_file = sys.argv[1]
    word = sys.argv[2]
    slur = sys.argv[3]
    model = sys.argv[4]
    temporal_viz(bad_words_file, word, slur, model)