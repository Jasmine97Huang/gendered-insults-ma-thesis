# Changing Semantics of Gendered Insults in Music Lyrics
This repo contains code and data for a paper submitted in partial fulfillment of the requirements for the Master of Arts degree in the Master of Arts in Computational Social Science
## Abstract
Popular culture has been heavily criticized for normalizing gendered insults and perpetuating gender stereotypes. Lyrics, powerful platforms for social change and artistic expressions, are particular salient venues to study the changing meanings of gendered insults and the cultural shifts underneath. Using neural network-based word representations, this article dynamically captures different uses of a gender slur in more than 1.6 million lyric sentences over time among different gender groups. First of its kind in both scale and approach, this analysis tracks the evolutions of the referents and connotations of gendered insults to highlight the progress, struggles, and limitations of the feminist movement in the popular culture space.
## Repo Descriptions 
| Name  | Type  | Description  |
|---|---|---|
| viz  | file | Visualizations & Plots |
| bert_mlm.sbatch | script | sbatch script to train BERT on cluster  |
| bert_utils.py | script | utility functions to retrieve embeddings |
| embeddings.ipynb | notebook | train, test models, clustering analysis and present visualization  |
| main.py | script | segment and preprocess data for MLM |
| model.py | script | Customize dataset class |
| preprocessing.ipynb | notebook | data merging, cleaning, record-linkage and prevalence analysis that includes counting/frequency plots |
| preprocessing.py | script | helper functions for preprocessing |
| train.py| script | BERT fine-tuning script |
| viz.py | script | visualize clusters by decades|
| viz.sbatch | script | sbatch script to visualize BERT on cluster  |
