import json
import pandas as pd
sentences = []
labels = []

#https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection?select=Sarcasm_Headlines_Dataset.json

def get_csv(file_path):
    for item in open(file_path, 'r'):
        item = json.loads(item)
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    data = pd.DataFrame({'headline':sentences, 'is_sarcastic':labels })
    train = data.iloc[:int(len(data)*0.9),:]
    test = data.iloc[int(len(data)*0.90):,:]
    
    return train, test
        

#get_csv('Sarcasm_Headlines_Dataset.json','train.csv')
train, test = get_csv('Sarcasm_Headlines_Dataset.json')
print(train.shape)
train.to_csv('data/train.csv',index=False)#, sep='\t')
test.to_csv('data/test.csv',index=False)#, sep='\t')


    
