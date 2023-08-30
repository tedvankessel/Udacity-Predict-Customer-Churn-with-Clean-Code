# library doc string
'''
Module:  churn_library.py

Purpose: Library Module for Customer Churn Data Analysis

Author : Theodore van Kessel

Date : 2023/08/28
'''

# Import libraries
import os
import warnings
import joblib
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report


# Filter out the specific warning fot rapid testing
warnings.filterwarnings("ignore", category=UserWarning,
                        message="Precision and F-score are ill-defined.*")

os.environ['QT_QPA_PLATFORM']='offscreen'

# Define default paths
DATA_PATH = './data/bank_data.csv'
EDA_IMAGE_PATH = './images/eda/'
RESULTS_IMAGE_PATH = './images/results/'
MODEL_PATH = './models/'

# setup logging configuration as shown in the course materials
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def import_data(pth):
    '''
    Purpose: Loads and returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    data_df = pd.read_csv(pth)

    return data_df

def perform_eda(data_df):
    '''
    perform EDA on data_df and saves figures to images folder
    input:
            data_df: pandas dataframe

    output:
            data_df: pandas dataframe
    '''
    # Copy DataFrame
    eda_df = data_df.copy(deep=True)

    # Convert Attrition_Flag categorical variable to Churn numerical variable
    # note: this is being done on a copy of the dataframe
    eda_df['Churn'] = eda_df["Attrition_Flag"].apply(
        lambda val: 0 if val=="Existing Customer" else 1)
    eda_df = eda_df.drop('Attrition_Flag', axis=1)

    # Set font size for labels and titles
    plt.rcParams.update({'font.size': 24})

    # Churn Histogram
    logging.info("INFO: Plotting Churn Histogram")
    plt.figure(figsize=(20, 10))
    eda_df['Churn'].hist()
    plt.xlabel('Churn')
    plt.ylabel('Frequency')
    plt.title('Churn Histogram')
    plt.savefig(fname = EDA_IMAGE_PATH + 'churn_histogram.png')
    plt.close()

    # Customer Age Histogram
    logging.info("INFO: Plotting Customer Age Histogram")
    plt.figure(figsize=(20, 10))
    eda_df['Customer_Age'].hist()
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Customer Age Histogram')
    plt.savefig(fname = EDA_IMAGE_PATH + 'customer_age_histogram.png')
    plt.close()

    # Marital Status Distribution
    logging.info("INFO: Plotting Marital Status Distribution")
    plt.figure(figsize=(20, 10))
    eda_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.xlabel('Marital Status')
    plt.ylabel('Normalized Frequency')
    plt.title('Marital Status Distribution')
    plt.savefig(fname = EDA_IMAGE_PATH + 'marital_status_distribution.png')
    plt.close()

    # Total Transaction Histogram
    logging.info("INFO: Plotting Total Transaction Histogram")
    plt.figure(figsize=(20, 10))
    sns.histplot(eda_df['Total_Trans_Ct'], kde=True)
    plt.xlabel('Total Transaction Count')
    plt.ylabel('Frequency')
    plt.title('Total Transaction Histogram')
    plt.savefig(fname = EDA_IMAGE_PATH + 'total_transaction_histogram.png')
    plt.close()

    # Heatmap
    plt.figure(figsize=(20, 10))

    # Select numeric columns for the correlation matrix
    numeric_columns = eda_df.select_dtypes(include=['number'])
    correlation_matrix = numeric_columns.corr()

    # Create the heatmap
    logging.info("INFO: Plotting Heatmap")
    plt.rcParams.update({'font.size': 14})
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.title('Correlation Heatmap')
    plt.savefig(fname = EDA_IMAGE_PATH + 'heatmap.png')
    plt.close()

    # Reset font size to default after plotting
    plt.rcParams.update({'font.size': 10})

    # Return dataframe
    return data_df

def encoder_helper(data_df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [argument that is
                be used for naming variables or index y column]

    output:
            data_df: pandas dataframe with new columns for training
    '''

    # Create a dictionary to store mapping of categories to numeric values
    category_mapping = {}

    # Loop through each categorical column
    logging.info("INFO: mapping categorical features")
    for column in category_lst:
        unique_categories = data_df[column].unique()  # Get unique categories in the column
        category_mapping[column] = {
            category: idx + 1 for idx, category in enumerate(unique_categories)}

    # Apply label encoding to each categorical column
    logging.info("INFO: apply mapping to features, reanme and drop old column")
    for column in category_lst:
        data_df[f'{column}_{response}'] = data_df[column].map(category_mapping[column])
        data_df.drop(column, axis=1, inplace=True)

    return data_df

def perform_feature_engineering(dataframe, response='Churn'):
    '''
    Converts remaining categorical features to numerical adding the response
    str prefix to new columns Then generate train and test datasets

    input:
                      dataframe: pandas dataframe
                      response: string of response name [argument that
                      is used for naming variables or index y column]

    output:
                      X_train: X training data
                      X_test: X testing data
                      y_train: y training data
                      y_test: y testing data
    '''
    logging.info("INFO: converting categorical variable Attrition_Flag to numerical variable Churn")
    dataframe['Churn'] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val=="Existing Customer" else 1)
    dataframe = dataframe.drop('Attrition_Flag', axis=1)

    # Collect categorical features to be encoded
    logging.info("INFO: Identify categorical features")
    cat_columns = dataframe.select_dtypes(include='object').columns.tolist()
    logging.info(f"INFO: categorical {cat_columns}")


    # Encode categorical features using mean of response variable on category
    dataframe = encoder_helper(dataframe, cat_columns, response='Churn')

    # print(dataframe.head(10))

    # create variables from dataframe, split and return train test data
    y = dataframe[response]
    X = dataframe.drop(response, axis=1)
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    logging.info(f"INFO: X_train shape: {X_train.shape}")
    logging.info(f"INFO: X_test shape:  {X_test.shape}")
    logging.info(f"INFO: y_train shape: {y_train.shape}")
    logging.info(f"INFO: y_test shape:  {y_test.shape}")

    return X_train, X_test, y_train, y_test

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # generate classification reports
    model_name = ['Random Forest', 'Logistic Regresson']
    y_train_models = [y_train_preds_rf, y_train_preds_lr]
    y_test_models = [y_test_preds_rf, y_test_preds_lr]
    name_image_save = ['rf_results', 'logistic_results']

    plt.close()  # close any open plots
    for model_index in range(len(model_name)):
        fig, ax = plt.subplots(figsize=(10, 10))

        plt.text(0.01, 1.25, f'{model_name[model_index]} Train', \
            fontsize=10, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(
            y_train, y_train_models[model_index])), fontsize=10, fontproperties='monospace')
        plt.text(0.01, 0.6, f'{model_name[model_index]} Test', \
            fontsize=10, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(
            y_test, y_test_models[model_index])), fontsize=10, fontproperties='monospace')

        plt.axis('off')
        plt.savefig(f'images/results/{name_image_save[model_index]}.png')
        plt.close(fig)


def feature_importance_plot(model, features, output_pth):
    '''
    Creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            features: pandas dataframe of X values
            output_pth: path to store the figure
    output:
             None
    '''
    # Feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort Feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Sorted feature importances
    names = [features.columns[i] for i in indices]

    # Create plot
    fig=plt.figure(figsize=(25, 15))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(features.shape[1]), importances[indices])

    # x-axis labels
    plt.xticks(range(features.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(fname=output_pth + 'feature_importances.png')
    plt.close(fig)

def train_models(X_train, X_test, y_train, y_test):
    '''
    Train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    logging.info("INFO: begin model training")
    # RandomForestClassifier and LogisticRegression
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Parameters for Grid Search
#     param_grid = {'n_estimators': [200, 500],
#                   'max_features': ['auto', 'sqrt'],
#                   'max_depth' : [4, 5, 100],
#                   'criterion' :['gini', 'entropy']}
    param_grid = {'n_estimators': [200, 500]}

    logging.info("INFO: begin grid search on random_forest classifier")
    # Grid Search and fit for RandomForestClassifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    logging.info("INFO: grid search complete")

    logging.info("INFO: begin training on logistic regression classifier")
    # LogisticRegression
    lrc = LogisticRegression(n_jobs=-1, max_iter=1000)
    lrc.fit(X_train, y_train)

    # Save best models
    logging.info("INFO: Save best models")
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Compute train and test predictions for RandomForestClassifier
    logging.info("INFO: Compute train and test predictions for RandomForestClassifier")
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf  = cv_rfc.best_estimator_.predict(X_test)

    # Compute train and test predictions for LogisticRegression
    logging.info("INFO: Compute train and test predictions for LogisticRegression")
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr  = lrc.predict(X_test)

    # Compute ROC curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()

    logging.info("INFO: Compute an save ROC curves")

    # Create a figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot ROC curve for LogisticRegression
    roc_display = RocCurveDisplay.from_estimator(lrc, X_test, y_test)
    roc_display.plot(ax=axes[0])

    # Plot ROC curve for RandomForestClassifier
    roc_display = RocCurveDisplay.from_estimator(cv_rfc.best_estimator_, X_test, y_test)
    roc_display.plot(ax=axes[1])

    # Save the figure and close it
    plt.savefig(fname='./images/results/roc_curve_result.png')
    # plt.show()
    plt.close(fig)

    # Compute results
    logging.info("INFO: Compute classification report")
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr,  y_test_preds_rf)

    # Compute feature importance
    logging.info("INFO: Compute feature importances")
    feature_importance_plot(model=cv_rfc,
                            features=X_test,
                            output_pth='./images/results/')

def main():
    '''
    Runs all the functions above to fully process the data   
    
    input:
            none
    output:
            results of the data procesing and model training in the res
    '''
    logging.info("INFO: Loading csv file into dataframe")
    df_data = import_data(DATA_PATH)

    logging.info("INFO: Performing and saving EDA plots")
    perform_eda(df_data)

    logging.info("INFO: Performing Feature Engineering")
    X_train, X_test, y_train, y_test = perform_feature_engineering(df_data, response='Churn')

    logging.info("INFO: Performing model training")
    train_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
