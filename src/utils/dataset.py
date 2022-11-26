import pandas as pd

from utils.download import dowanlod


def prepare_dataset():
    df = pd.read_json(dataset_path+'/Enron_train.jsonl', lines=True)

    df = df[['text','label']]

    # change <2 = non coherence
    df.loc[df['label'] <=2, 'label'] = 0
    # change >0 = coherence
    df.loc[df['label'] >0, 'label'] = 1

    df.label.value_counts()

    df['text'] = df['text'].replace(r'\s+|\\n', ' ', regex=True) 

    df.to_csv(dataset_path + '/dataset.csv', header=True, index=False)

