# ST-DeepGait Code Repo

## How to Run
To run ST-DeepGait, run:

 `python train.py --data_file <path_to_data> --label_file <path_to_label_file> --indices_file <path_to_indices_file>`  
 
 The number of epochs used and learning rate can also be changed using the `--num_epochs` and `--learning_rate` flags respectively.
 
 We reccommend training using a GPU-enabled machine. 

## Results

### Frozen Models
After training is complete, `save_dir` will contain the frozen models for the 10 most recent training epochs, and `best_model` will contain the frozen models for the 4 "best" models, as determined by the validation error calculated at the end of the epoch.

### Embeddings
In addition, when training is complete, `train.py` will write out `full_embeddings.npy` and `best_embeddings.npy` which contain 128-dimensional *embedding layer* predictions for each of the runs. These embeddings use the same order as the original data input and labels file. 

To test KNN Accuracy on the embeddings, read `best_embeddings.npy` or `full_embeddings.npy` into a Python environment, along with the original labels file. To perform a train-test split, read the indices file you used during training, and then call the `split_data_existing` function from `train.py`, which will split the dataset based on the indices file you chose. 

Once split, you can feed the training set and testing set to a KNN classifier (`sklearn.neighbors.KNeighborsClassifier` for example) to evaluate KNN performance.



