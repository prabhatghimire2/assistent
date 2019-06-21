import sqlite3
import pandas as pd

timeframes = ['machineLearning']

for timeframe in timeframes:
    connection = sqlite3.connect('{}.db'.format(timeframe))
    cursor = connection.cursor()
    limit = 500
    last_unix = 0
    cur_length = limit
    counter = 0
    test_done = False
    while cur_length == limit:
        df = pd.read_sql("SELECT * from chatbot WHERE unix > '{}' AND parent NOT NULL AND score > 0 ORDER BY unix ASC LIMIT '{}'".format(last_unix, limit), connection)
        print(df)
        last_unix = df.tail(1)['unix'].values[0]
        cur_length = len(df)
        if not test_done:
            with open('test.from', 'a', encoding='utf8') as file:
                for content in df['parent'].values:
                    file.write(content+'\n')
            with open('test.to', 'a', encoding='utf8') as file:
                for content in df['comment'].values:
                    file.write(content+'\n')
            test_done = True
        else :
            with open('train.from', 'a', encoding='utf8') as file:
                for content in df['parent'].values:
                    file.write(content+'\n')
            with open('train.to', 'a', encoding='utf8') as file:
                for content in df['comment'].values:
                    file.write(content+'\n')

        counter +=1
        if counter%20==0:
            print(counter*limit,'row completed so far')

