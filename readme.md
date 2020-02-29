## Reddit Rep 2

New ideas for measuring reputation based on graph neaural nets.

An example of how to use the tools here is present in `main.py`.

Reddit data is avaliable at https://files.pushshift.io/reddit/comments/

## Setting up GloVe vectors

The setup for GloVe embeddings follows from this tutorial: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

1. Download the word vectors glove.6B.zip from: https://nlp.stanford.edu/projects/glove/
2. Extract the zip folder contents into `machine_learning/glove`
3. run `cd machine_learning && python create_glove_embedding.py`. This will process the word vectors and save the results into pkl files.


## Testing Predictions

1. if not done already, run `pip install -r requirements.txt`
2. run `python process_trees.py`. This creates a pickle file of 100 processed trees
3. cd to `machine_learning` and run `python glove_gat.py`. This trains the model on the pickle file just created
