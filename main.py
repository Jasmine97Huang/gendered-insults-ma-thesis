import preprocessing
import model
import sys 

if __name__ == "__main__":
    file_path = sys.argv[1]
    print(f"Segmenting file {file_path}...")
    preprocessing.segment_lyrics(file_path)
    print("Tokenizing file...")
    tokenized = preprocessing.tokenize("output/segmented.csv")
    print("Setting up MLM task architecture...")
    preprocessing.set_up_mlm(tokenized)
