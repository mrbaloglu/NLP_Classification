# NLP_Classification
Some implementations for NLP classification tasks.

## Datasets
### RT Polarity Data
This rating inference dataset is a sentiment classification dataset, containing 5,331 positive and 5,331 negative processed sentences from Rotten Tomatoes movie reviews. On average, these reviews consist of 21 words. Collected from https://www.cs.cornell.edu/people/pabo/movie-review-data/ and converted to a pandas dataframe.

### Yahoo Answers Data
The Yahoo! Answers topic classification dataset is constructed using 10 largest main categories. Each class contains 140,000 training samples and 6,000 testing samples. Therefore, the total number of training samples is 1,400,000 and testing samples 60,000 in this dataset. From all the answers and other meta-information, we only used the best answer content and the main category information. Collected from https://www.kaggle.com/datasets/soumikrakshit/yahoo-answers-dataset. 

## Code
### PyTorch Models (baseline_models.py)
Implementations of stacked transformer encoders, recurrent neural networks (with lstm and gru), and pretrained BERT model with PyTorch and Hugging Face's transformers library.

### Generic PyTorch Trainer (pytorch_trainer.py)
An implementation for a pytorch generic trainer class, to have an easy to use scheme for pytorch models.

### PyTorch Datasets (pytorch_datasets.py)
Implementations for pytorch datasets to convert pandas dataframes or numpy array into an appropriate format to be batched through pytorch's dataloaders.


