from preprocessing import *
import pandas as pd

if __name__ == '__main__':

    """
    # preprocessing the data
    data = pd.read_csv("rt-polarity-full.csv")
    data.columns = ['label', 'review']
    data_prc = process_df_texts(data, ["review"])
    data_tkn = tokenize_data(data, ["review"], preprocess=True)
    data_sqn = pad_tokenized_data(data_tkn, ["review_tokenized"])

    # save the processed data in a csv file
    data_sqn.to_csv("rt-polarity-processed.csv")
    """


    data = pd.read_csv("rt-polarity-processed.csv")

    



