'''
Module:  churn_script_logging_and_tests.py

Purpose: Test Library Module for Customer Churn Data Analysis

Author : Theodore van Kessel

Date : 2023/08/28
'''
import os
import logging
import pandas as pd
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_script_logging_and_tests.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# global variables
dataframe = pd.DataFrame()
X_train = [] 
X_test = []
y_train = []
y_test = []

def test_import_data():
    '''
    Test import_data() function from the churn_library module
    '''
    global dataframe
    # Test if the CSV data file is available
    try:
        dataframe = cls.import_data("./data/bank_data.csv")
        logging.info("SUCCESS: Testing import_data file loaded")
    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_data: The file not found")
        raise err

    # Test the dataframe to see if it contains actual data
    try:
        assert dataframe.shape[0] > 0      
        assert dataframe.shape[1] > 0     
        logging.info('INFO: Rows: %d\tColumns: %d', dataframe.shape[0], dataframe.shape[1]) 
    except AssertionError as err:
        logging.error("ERROR: Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_perform_eda():
    '''
    Test perform_eda() function from the churn_library module
    '''
    global dataframe
    
    # List of file paths produced in this function
    file_paths = [
        "./images/eda/churn_histogram.png",
        "./images/eda/customer_age_histogram.png",
        "./images/eda/marital_status_distribution.png",
        "./images/eda/total_transaction_histogram.png",
        "./images/eda/heatmap.png"
    ]

    # Delete pre-existing files
    logging.info("INFO: Deleting pre-existing files")
    for file_path in file_paths:
        try:
            os.remove(file_path)
            logging.info(f"INFO: Deleted pre-existing file {file_path}")
        except OSError as e:
            # Handle exceptions, like "FileNotFoundError" or "PermissionError"
            logging.info(f"INFO: pre-existing file {file_path}: {e}")
        
    # Try to perform eda 
    try:
        dataframe = cls.perform_eda(dataframe)
        logging.info("SUCCESS: perform_eda ran")
    except KeyError as err:
        logging.error('ERROR: perform_eda fail', err.args[0])
        raise err

    # Check each file in list
    for file_path in file_paths:
        if os.path.isfile(file_path):
            logging.info(f'SUCCESS: File {file_path} was found')
        else:
            logging.error(f'ERROR: File {file_path} not found')

def test_encoder_helper():
    '''
    test encoder_helper function
    '''
    global dataframe
     
    # in order to test the encoder_helper function we need to create the necessary components to pass in
    df_copy = dataframe.copy()
    category_lst = [
        'Gender',
        ]
    response = 'Churn'
    
    # test the encoder_helper function
    try:
        df_copy = cls.encoder_helper(df_copy, category_lst, response)
        logging.info("SUCCESS: encoder_helper ran")
    except KeyError as err:
        logging.error('ERROR: encoder_helper fail', err.args[0])
        raise err
    
    try:
        assert('Gender_Churn' in df_copy.columns)   
        logging.info('SUCCESS: Confirmed encoder_helper makes the correct modification to the dataframe') 
    except AssertionError as err:
        logging.error("ERROR: encoder_helper did not make the correct modification to the dataframe")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    global dataframe, X_train, X_test, y_train, y_test
    
    logging.info("INFO: testing perform_feature_engineering")
    try:
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(dataframe, response='Churn')
        logging.info("SUCCESS: perform_feature_engineeringr ran")
    except KeyError as err:
        logging.error('ERROR: perform_feature_engineeringr fail', err.args[0])
        raise err
        
    # check results        
    conditions = [
        X_train.shape[0] > 0,
        X_test.shape[0] > 0,
        y_train.shape[0] > 0,
        y_test.shape[0] > 0,
        X_train.shape[1] == X_test.shape[1],
        X_train.shape[0] == y_train.shape[0],
        X_test.shape[0] == y_test.shape[0]
    ]
    
    failed_conditions = [index for index, condition in enumerate(conditions) if not condition]
    
    if failed_conditions:
        failed_messages = [
            "X_train.shape[0] is not greater than 0",
            "X_test.shape[0] is not greater than 0",
            "y_train.shape[0] is not greater than 0",
            "y_test.shape[0] is not greater than 0",
            "X_train.shape[1] is not equal to X_test.shape[1]",
            "X_train.shape[0] is not equal to y_train.shape[0]",
            "X_test.shape[0] is not equal to y_test.shape[0]"
        ]
        failed_msgs = [failed_messages[index] for index in failed_conditions]
        raise AssertionError("\n".join(failed_msgs))
    else:
        logging.info("SUCCESS: All conditions on data passed successfully")

    


def test_train_models():
    '''
    test train_models
    '''
    
    global dataframe, X_train, X_test, y_train, y_test
    
    logging.info("INFO: testing train_models")
    
    # List of file paths produced in this function
    file_paths = [
        "./models/logistic_model.pkl",
        "./models/rfc_model.pkl",
        "./images/results/roc_curve_result.png",
        "./images/results/rf_results.png",
        "./images/results/logistic_results.png",
        "./images/results/feature_importances.png",
        ]

    # Delete pre-existing files
    logging.info("INFO: Deleting pre-existing files")
    for file_path in file_paths:
        try:
            os.remove(file_path)
            logging.info(f"INFO: Deleted pre-existing file {file_path}")
        except OSError as e:
            # Handle exceptions, like "FileNotFoundError" or "PermissionError"
            logging.info(f"INFO: pre-existing file {file_path}: {e}")
    
    # run the train_models function
    logging.info("INFO: testing train_models function")
    try:
        cls.train_models(X_train, X_test, y_train, y_test)
        logging.info("SUCCESS: train_models ran")
    except KeyError as err:
        logging.error('ERROR: train_models fail', err.args[0])
        raise err
        
    logging.info("INFO: testing train_models produced files") 
    # Check each file in list
    for file_path in file_paths:
        if os.path.isfile(file_path):
            logging.info(f'SUCCESS: File {file_path} was found')
        else:
            logging.error(f'ERROR: File {file_path} not found')
    

def main():
    """
    If running as main program, execute the test functions of the churn_library.py 
    """
    global dataframe
    
    # test the import function
    logging.info("INFO: Begin testing import_data")
    test_import_data()
    
    # test perform_eda function
    test_perform_eda()
    
    # test encoder_helper function
    test_encoder_helper()
    
    # test perform_feature_engineering function
    test_perform_feature_engineering()
    
    # test train_models function
    test_train_models()

if __name__ == "__main__":
    main()