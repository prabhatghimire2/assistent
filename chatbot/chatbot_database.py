import sqlite3
import json
from datetime import datetime

timeframe = 'machineLearning'
sql_transaction = []

connection = sqlite3.connect('{}.db'.format(timeframe))
cursor = connection.cursor()

def create_table():
    cursor.execute("""CREATE TABLE IF NOT EXISTS
        chatbot(parent_id TEXT,
        comment_id TEXT,
        parent TEXT,                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        comment TEXT,
        subreddit TEXT,
        unix INT,
        score INT)""" )


def format_data(data):
    data = data.replace('\n', ' newlinechar ').replace('\r',' newlinechar ').replace('"',"'")
    return data

def find_existing_score(pid):
    try:
        sql = """SELECT score FROM chatbot WHERE parent_id='{}' LIMIT 1""".format(pid)
        cursor.execute(sql)
        result = cursor.fetchone()
        if result != None:
            return result[0]
        else : return False
    except Exception as e:
        print('parent_id :', e)
        return False


def find_parent(pid):
    try:
        sql = """SELECT comment FROM chatbot WHERE comment_id='{}' LIMIT 1""".format(pid)
        cursor.execute(sql)
        result = cursor.fetchone()
        if result != None:
            return result[0]
        else : return False
    except Exception as e:
        print('parent_id :', e)
        return False


def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        cursor.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                cursor.execute(s)
            except Exception as e:
                print('transaction error : ', e)
        connection.commit()
        sql_transaction = []



def sql_insert_replace_comment(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = """UPDATE chatbot SET parent_id = '{}', comment_id = '{}', parent = '{}', comment = '{}', subreddit = '{}', unix = '{}', score = '{}' WHERE parent_id ='{}';""".format(
            parentid, commentid, parent, comment, subreddit, int(time), score, parentid)
        transaction_bldr(sql)
    except Exception as e:
        print('s-UPDATE ',str(e))


def sql_insert_has_parent(commentid,parentid,parent,comment,subreddit,time,score):
    try:
        sql = """INSERT INTO chatbot (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parentid, commentid, parent, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s-PARENT',str(e))


def sql_insert_no_parent(commentid,parentid,comment,subreddit,time,score):
    try:
        sql = """INSERT INTO chatbot(parent_id, comment_id, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}",{},{});""".format(parentid, commentid, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s-NO_PARENT',str(e))




def acceptable(data):
    if len(data.split(' '))> 50 or len(data) < 1:
        return False
    elif len(data)> 1000 :
        return False
    elif data == '[deleted]' or data == '[removed]' :
        return False
    else :
        return True


if __name__ == '__main__':
    create_table()
    row_counter = 0
    paired_rows = 0

    with open('RC_2015-01', buffering=1000) as file:
        for row in file:
            row_counter +=1
            row = json.loads(row)
            parent_id = row['parent_id']
            body = format_data(row['body'])
            created_utc = row['created_utc']
            score = row['score']
            subreddit = row['subreddit']
            parent_data = find_parent(parent_id)
            comment_id = row['name']



            if score >= 2:
                if acceptable(body):
                    existing_comment_score = find_existing_score(parent_id)
                    if existing_comment_score:
                        if score > existing_comment_score:
                            sql_insert_replace_comment(comment_id,
                                parent_id,
                                parent_data,
                                body,
                                subreddit,
                                created_utc,
                                score)
                    else:
                        if parent_data:
                            sql_insert_has_parent(comment_id,
                                parent_id,
                                parent_data,
                                body,
                                subreddit,
                                created_utc,
                                score)
                            paired_rows += 1
                        else:
                            sql_insert_no_parent(comment_id,
                                parent_id,
                                body,
                                subreddit,
                                created_utc,
                                score)
        if row_counter % 10000 == 0:
            print(f'Total row read : {row_counter}, Paired_rows: {paired_rows}, Time : {str(datetime.now())}')
