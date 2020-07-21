import pandas as pd
import pyodbc
import datetime
import re
import logging
import os.path
from datetime import datetime
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
from ftfy import fix_text
import csv
import shutil
import time


def initialize_logger(output_dir, log_name):
    """
    :param output_dir: log file directory
    :param log_name: log file name
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(output_dir, log_name + "_Error.log"), "w", encoding=None, delay="true")
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(output_dir, log_name + "_All.log"), "w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def exec_sql(server, database, sql_procedure, params):
    """
    :param server: The name of the server name (JZNVCS)
    :param database: The name of the database (VCS)
    :param sql_procedure: The name of the sql procedure
    :param params: The parameters of the sql procedure
    :return: Nothing, otherwise sql error
    """
    conn = pyodbc.connect("Driver={SQL Server};"
                          "Server=" + server + ";"
                          "Database=" + database + ";"
                          "Trusted_Connection=yes;")
    conn.autocommit = True
    cursor = conn.cursor()
    sql_procedure = 'EXEC ' + sql_procedure

    try:
        if params == '':
            cursor.execute(sql_procedure)
        else:
            cursor.execute(sql_procedure, params)

        rows = cursor.fetchall()
        while rows:
            if cursor.nextset():
                rows = cursor.fetchall()
            else:
                rows = None
        cursor.commit()
        logging.info(database + ' - SQL procedure executed successfully!')
    except Exception as e:
        if str(e) == 'No results.  Previous SQL was not a query.':
            logging.info(database + ' - SQL procedure executed successfully! - No results')
            cursor.commit()
        else:
            logging.error(str(e))
    finally:
        conn.close


def clean_text(text):
    """
    :param text: Raw text to clean
    :return: Lowercase text without special characters
    """
    replace_with_spaces = re.compile(r'[/(){}\[\]\|@,;]')
    remove_bad_symbols = re.compile('[^0-9a-z #+_]')
    english_stopwords = set(stopwords.words('english'))

    text = BeautifulSoup(text, "lxml").text
    text = text.lower()
    text = replace_with_spaces.sub(' ', text)
    text = remove_bad_symbols.sub('', text)
    text = ' '.join(word for word in text.split() if word not in english_stopwords)
    return text


def get_n_grams(string, n=3):
    """
    :param string: String to be checked
    :param n: Length of the grams
    :return: All grams
    """
    string = fix_text(string)
    string = string.encode("ascii", errors="ignore").decode()
    string = string.lower()
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title()
    string = re.sub(' +', ' ', string).strip()
    string = ' ' + string + ' '
    string = re.sub(r'[,-./]|\sBD', r'', string)
    n_grams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in n_grams]


def get_nearest_neighbor(engine, algorithm, query):
    """
    :param engine: Engine object
    :param algorithm: algorithm
    :param query: Text to transform
    :return: Distance(s) and index(es) of the text
    """
    transformed_query = engine.transform(query)
    distances, indices = algorithm.kneighbors(transformed_query)
    return distances, indices


def result_category(similarity, confidence):
    """
    :param similarity: Similarity level (1 -100)
    :param confidence: confidence level (0 - 10)
    :return: Category name
    """

    if similarity >= 87 and confidence <= 1.0:
        return 'UPDATE'
    else:
        return 'LOW ACCURACY'


def main():

    export_dir = r'\\JZNVCS\VCS\Client_Exports\Jzanus\Machine_Learning\Denials_Prediction'
    import_dir = r'\\JZNVCS\VCS\Client_Imports\Jzanus\Machine_Learning\Denials_Prediction'
    train_dir = r'\\JZNVCS\VCS\Client_Exports\Jzanus\Machine_Learning\Denials_Prediction\Train_Data'
    test_dir = r'\\JZNVCS\VCS\Client_Exports\Jzanus\Machine_Learning\Denials_Prediction\Test_Data'
    train_file = ''
    log_dir = r'\\JZNVCS\VCS\Log_file'
    sql_server = 'JZNVCS'
    sql_database = 'VCS'

    # Start logging
    now = datetime.now()
    log_file = 'Denials_Prediction_' + now.strftime("%m%d%Y%H%M%S")
    initialize_logger(log_dir, log_file)

    # Check previous unprocessed files
    file_list = [f for f in os.listdir(export_dir) if f.endswith('.csv')]
    for f in file_list:
        if f.startswith('DP_Test'):
            shutil.move(os.path.join(export_dir, f), os.path.join(test_dir, f))
        else:
            shutil.move(os.path.join(export_dir, f), os.path.join(train_dir, f))

    # Create Files
    exec_sql(sql_server, sql_database, 'proc_ML_DenialsPrediction_Export @tcType=?', 'TRAIN')
    time.sleep(6)

    # Create Files
    exec_sql(sql_server, sql_database, 'proc_ML_DenialsPrediction_Export', '')
    time.sleep(6)

    # Find test files to process
    files_founds = [f for f in os.listdir(export_dir) if f.endswith('.csv') and f.startswith('DP_Test')]
    logging.info('{} test file(s) created!'.format(len(files_founds)))

    # Find latest training data set
    train_sets = [f for f in os.listdir(export_dir) if f.endswith('.csv') and f.startswith('DP_Train')]
    logging.info('{} train file(s) found!'.format(len(train_sets)))

    for data_set in train_sets:
        train_file = data_set

    # Clean train data
    train_data = pd.read_csv(os.path.join(export_dir, train_file), sep='|', encoding='unicode_escape')
    train_data['ReasonDescription'] = train_data['ReasonDescription'].apply(clean_text)

    # Predict the denials of each file
    for file in files_founds:

        try:
            results_file = 'Results_' + file
            excel_report = 'Results_' + file.replace('.csv', '.xlsx')

            test_data = pd.read_csv(os.path.join(export_dir, file), sep='|', encoding='unicode_escape')

            # Start prediction process
            test_data['ReasonDescription'] = test_data['ReasonDescription'].apply(clean_text)

            logging.info('Vectorization of the data...')

            historical_denial = train_data['ReasonDescription']
            engine = TfidfVectorizer(min_df=1, analyzer=get_n_grams, lowercase=False)
            model = engine.fit_transform(historical_denial)

            logging.info('Vectorization completed.')

            algorithm = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(model)
            new_denial = test_data['ReasonDescription']
            trans_denial = test_data['DenialsPk']

            t1 = time.time()
            logging.info('Getting nearest n...')
            distances, indices = get_nearest_neighbor(engine, algorithm, new_denial)
            t = time.time() - t1
            logging.info('Completed in: ' + str(t))

            logging.info('Finding matches...')
            new_denial = list(new_denial)
            trans_denial = list(trans_denial)

            matches = []
            for i, j in enumerate(indices):
                temp = [trans_denial[i],
                        new_denial[i],
                        train_data.values[j][0][1],
                        train_data.values[j][0][2],
                        round(distances[i][0], 2),
                        fuzz.token_sort_ratio(new_denial[i], train_data.values[j][0][1])]

                matches.append(temp)

            logging.info('Building results data set...')
            results = pd.DataFrame(matches, columns=['DenialsPk',
                                                     'NewDenial',
                                                     'MatchedDenial',
                                                     'MatchedProcess',
                                                     'MatchConfidence',
                                                     'SimilarityRatio'])

            results['ResultCategory'] = results.apply(lambda x: result_category(x['SimilarityRatio'],
                                                                                x['MatchConfidence']),
                                                      axis=1)
            logging.info('Done')

            # Create results
            results.to_csv(os.path.join(import_dir, results_file), sep="|", index=False, quoting=csv.QUOTE_NONE)
            writer = pd.ExcelWriter(os.path.join(import_dir, excel_report), engine='xlsxwriter')
            results.to_excel(writer, sheet_name='Sheet1', index=False)
            writer.save()

            logging.info(results_file + ' created!')
            logging.info(excel_report + ' created!')

            # Load results
            exec_sql(sql_server, sql_database, 'proc_ML_DenialsPrediction_Import @tcFileName=?', results_file)

            # Move file to test data folder
            shutil.move(os.path.join(export_dir, file), os.path.join(test_dir, file))
            shutil.move(os.path.join(export_dir, train_file), os.path.join(train_dir, train_file))
            logging.info(file + ' has been process successfully!')

        except Exception as e:
            logging.error(str(e))


if __name__ == '__main__':
    main()
