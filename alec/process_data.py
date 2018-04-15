"""
This script will preprocess the text and convert the documents into a TF-IDF matrix.
"""
import os
import re
import pickle
import sqlite3 as sqlite
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

DATADIR = '../data/'
DB = os.path.join(DATADIR, 'Reference Standard', 'radiology_reports.sqlite')
TABLE_NAMES = ('training_notes', 'testing_notes')


def preprocess_human(report):
    """Preprocesses a report for annotation
    Features:
        - Insert termination points before Headers
        - remove new page headers
        - replace exception words
        - delete ------- ???
        - delete [**3432]
        - delete excess whitespaces
        """

    #1) delete new page header
    header_exists = re.search("\(Over\)",report)
    while header_exists:
        #find the start of the word 'over'
        re_over = re.compile(r"""\(Over\)""")
        header_over = re_over.search(report)
        header_start = header_over.start()
        #find end of the word 'cont'
        re_cont = re.compile(r"""\(Cont\)""")
        header_cont = re_cont.search(report)
        header_end = header_cont.end()
        #return report[header_end+1]
        #return header_start, header_end
        report = report[:header_start] + report[header_end:]
        header_exists = re.search("\(Over\)",report)
    #0) add a ~ before a heading
    #header_pattern = re.compile(r"""[A-Z])
    headers = ["UNDERLYING MEDICAL CONDITION:", "REASON FOR THIS EXAMINATION:","Reason:","REASON:", "INDICATION:",
               "TECHNIQUE:","WITH IV CONTRAST:","CT OF","IMPRESSION:","WET READ:", "Admitting Diagnosis",
               "COMPARISON:", "FINDINGS:","ADDENDUM:","PFI:","ADDENDUM","PROVISIONAL FINDINGS IMPRESSION","FINAL REPORT"]

    for h in headers:
        report = re.sub(h, "~ "+h,report)

    #Replace exception words
    report = re.sub('d\.r\.|dr\.|Dr\.|DR\.','DR',report)
    report = re.sub('EG\.|e\.g\.|eg\.','eg,',report)
    report = re.sub('Mr\.|MR\.|mr\.','MR',report)
    report = re.sub('Mrs\.|MRS\.|mrs\.','MRS',report)
    report = re.sub('Ms\.|MS\.|ms\.','MS',report)
    report = re.sub('M\.D\.|MD\.|M\.d\.|m\.d\.','MD',report)
    report = re.sub('\d{1,}\.','-',report)
    #4) delete deidentified data fields[** **]
    #report = re.sub('\[\*\*[\w\-\(\)]{1,}\*\*\ ]','',report)
    report = re.sub('\[\*\*[^\*]{1,}\*\*\]','',report)
    report = re.sub('Clip #','',report)

    # delete times
    report = re.sub('(\d{1,2}:\d{2} )((AM)|(am)|(A.M.)|(a.m.)|(PM)|(pm)|(P.M.)|(p.m.))','',report)

    #2) delete excess whitespaces
    report = re.sub('[\n]{2,}','\n',report)
    report = re.sub('[\t]{2,}','\t',report)
    report = re.sub('[ ]{3,}',' ',report)

    #3) delete ----
    report = re.sub('[_]{5,}', '\n', report)


    return report


def preprocess(text):
    text = text.lower()
    return text


def transform_notes(notes):
    vectorizer = CountVectorizer(notes, min_df=0.1, lowercase=True, stop_words='english',
                                 ngram_range=(1, 3))
    X = vectorizer.fit_transform(notes)
    return X, vectorizer


def main():
    conn = sqlite.connect(DB)
    train_notes = pd.read_sql("SELECT * FROM training_notes;", conn)
    conn.close()
    print(train_notes.head())
    X = list(train_notes.text)
    y = list(train_notes.doc_class)
    X, vectorizer = transform_notes(X)

    print(X.shape)

    with open(os.path.join(DATADIR, 'train_data.pkl'), 'wb') as f:
        pickle.dump((X, y), f)

    with open(os.path.join(DATADIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)





if __name__ == '__main__':
    main()
