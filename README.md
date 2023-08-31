# Udacity-Predict-Customer-Churn-with-Clean-Code

## Project Description
### From the course documentation:

In this project, you will implement your learnings to identify credit card customers that are most likely to churn. The completed project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

## Code Quality tests

Autopep8 and pylint were run on both the **churn_library.py** and **churn_library.py** files as prescribed in the guide
and the results are shown in this github in the files:

**autopep8 and pylint results for churn_library.txt**

**autopep8 and pylint results for churn_script_logging_and_tests.txt**

Pylint results for both files were greater than **0.8**

## Files and data description
Overview of the files and data present in the root directory. 

	├── churn_library.py        ------------------------> functions that implement the churn model
	├── churn_notebook.ipynb    ------------------------> Original Udacity starter Jupyter notebook from churn model 
	├── churn_script_logging_and_tests.py  -------------> unit test functions to test churn_library functions
	├── data
	│   └── bank_data.csv   ----------------------------> data in csv format
	├── images
	│   ├── eda 
	│   │   ├── churn_histogram.png  -------------------> churn hystogram image
	│   │   ├── customer_age_histogram.png  ------------> customer age hystogram image
	│   │   ├── heatmap.png  ------> heatmap image
	│   │   ├── marital_status_hystogram.png  ----------> marital status hystogram image
	│   │   └── total_transaction_distribution.png  ----> total transaction distribution image
	│   └── results
	│       ├── feature_importances.png  ---------------> feature importances image
	│       ├── logistic_results.png  ------------------> logistic model results image
	│       ├── rf_results.png  ------------------------> random forest results image
	│       └── roc_curve_result.png  ------------------> roc curve result image
	├── logs
	│   └── churn_library.log  -------------------------> test logging
	├── models
	│   ├── logistic_model.pkl -------------------------> logistic model file
	│   └── rfc_model.pkl ------------------------------> random forest model file
	│
	├── README.md
	├── requirements_py3.6.txt

## Running Files
There are 2 operable files in this project: **churn_library.py** and **churn_library.py**
The churn_library.py file implements a set of functions to process the 	**bank_data.csv** file:

	Load and explore the dataset composed of over 10k samples (EDA)
	Prepare data for training (feature engineering resulting into 19 features)
	Train two classification models (sklearn random forest and logistic regression)
	Identify most important features influencing the predictions and visualize their impact using SHAP library
	Save best models with their performance metrics

 To run the files it is necessary to import the required files. These include:
 
 This file can be run from the command line or imported as a class and run with another program
 
 	Run: python churn_library.py 
 
 	run: python_script_logging_and_tests.py


## Code
The code for this project is fully embodied in the **churn_library.py** and **churn_library.py** files in this Github repository.

### The python version used in this code is:

	Python version
	3.7.6 (default, Jan  8 2020, 20:23:39) [MSC v.1916 64 bit (AMD64)]
	Version info.
	sys.version_info(major=3, minor=7, micro=6, releaselevel='final', serial=0)

>See below for more details on environment etc. 

## Datasets
The following datasets were recommended by Udacity for use in this project:

**bank_data.csv**	Provided by Udacity as part of the course project data.
It is my understanding that this is publically provided for use in this project

    
## Acknowledgments and Sources of Code and Data:

    >Udacity project documents
    >README-Template - https://gist.github.com/PurpleBooth/109311bb0361f32d87a2
    >ChatGPT
    >Udacity GPT
    >Google
    >https://github.com/anuraglahon16/Project-Predict-Customer-Churn-with-clean-code-
    >https://github.com/LaurentVeyssier/Udacity-Predict-Customer-Churn-with-Clean-Code
    >https://github.com/I-Akrout/Predict-customer-churn-clean-code
    >https://github.com/luizomatias/predict-customer-churn-clean-code
    >https://github.com/ibrahim-sheriff/Predict-Customer-Churn-with-Clean-Code


### Installing
The jupyter notebook file can be operated with the standard Jupyter Notebook software.
As in all projects, it is recommended to set up an environment with the required packages. These include:

	ipython=8.12.2=pypi_0
	numpy=1.24.3=py38h1d91fd2_0
	pandas=2.0.1=py38hf08cf0d_0
	matplotlib=3.7.1=pypi_0
	scikit-learn=1.3.0=py38h763eb3e_0
  torch 1.4
  torchvision 0.8

 
 ### NOTE: an **environment.yml file** can be found in this Github that contains all the packages used in this project environment.
 
 ## License
This project is licensed under the MIT License  License - see the LICENSE.md file for details

## Built With
	Anaconda
	Eclipse IDE for Developers (used for checking code)
	Jupyter Notebook
## Author
**Theodore van Kessel** 


