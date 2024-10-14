import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

final_df_list = []

def process_chunk(chunk, cv):
    ques_df = chunk[['question1', 'question2']]
    questions = list(ques_df['question1']) + list(ques_df['question2'])
    
    q1_arr, q2_arr = np.vsplit(cv.transform(questions).toarray(), 2)
    
    temp_df1 = pd.DataFrame(q1_arr, index=ques_df.index)
    temp_df2 = pd.DataFrame(q2_arr, index=ques_df.index)
    temp_df = pd.concat([temp_df1, temp_df2], axis=1)
    
    final_df_chunk = pd.concat([chunk.drop(columns=['id', 'qid1', 'qid2', 'question1', 'question2']), temp_df], axis=1)
    
    return final_df_chunk

chunk_size = 10000
csv_file = 'updated_train.csv'
reader = pd.read_csv(csv_file, chunksize=chunk_size)


ques_df_full = pd.read_csv(csv_file)[['question1', 'question2']]
questions_full = list(ques_df_full['question1']) + list(ques_df_full['question2'])

cv = CountVectorizer(max_features=3000)
cv.fit(questions_full)

for chunk in reader:
    processed_chunk = process_chunk(chunk, cv)
    final_df_list.append(processed_chunk)

final_df = pd.concat(final_df_list, ignore_index=True)

print(final_df.shape)
print(final_df.head())

final_df.to_csv('data_new.csv', index=False)
