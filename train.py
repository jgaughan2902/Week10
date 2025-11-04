import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pickle

def get_coffee_data():
    '''
    Retrieves the coffee data from the url.

    Parameters:
    No input values.

    Return value:
    df_coffee (pd.DataFrame): A dataframe produced
    from the data within the url.
    '''
    # Read the raw coffee csv from the url.
    df_coffee = pd.read_csv("https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv")

    # Drop duplicate instances.
    df_coffee.drop_duplicates(subset = 'desc_1', inplace = True)

    # Drop NA values.
    df_coffee.dropna(subset = ['desc_1', 'roast', 'loc_country'], inplace = True)

    return df_coffee

def fit_linear_regression():
    '''
    Fits a linear regression with 100g_USD as the
    lone predictor and rating as the response.

    Parameters:
    No input values.

    Return value:
    No return value but it does produce a pickle file
    containing the trained model.
    '''
    # Establish the data set.
    df_coffee = get_coffee_data()

    # Do a train, test set split using 80/20.
    df_train, df_test = train_test_split(df_coffee, test_size = 0.2, random_state = 42)

    # The only feature (predictor) is defined as 100g_USD.
    features = ['100g_USD']

    # Only the features in the train set.
    X = df_train[features]

    # Only the response in the train set.
    y = df_train['rating']

    # Initiate a LinearRegression object.
    lm = LinearRegression()

    # Fit the model using the features and response.
    lm.fit(X, y)

    # Create a pickle file with the model.
    try:
        with open('model_1.pickle', 'wb') as file:
            pickle.dump(lm, file)
    except Exception as e:
        print(f'An error occured')


def roast_category():
    '''
    Maps all categorical roast values to valid
    numerical inputs for the decision tree regressor model.

    Parameters:
    No input values.

    Return value:
    df_coffee (pd.DataFrame): A data frame that contains the
    coffee data but adds an additional column containing the
    recoded roast category column.
    '''
    # Pull the original data set.
    df_coffee = get_coffee_data()

    # Define the mapping for each category of roast.
    roast_map = {'Light' : 1, 'Medium-Light' : 2, 'Medium' : 3, 'Dark' : 4}

    # Define a new column with the new recoded roast categories.
    df_coffee['roast_cat'] = df_coffee['roast'].map(roast_map)

    # Drop any NAs although this isn't necessary for this assignment.
    df_coffee.dropna(subset = ['roast_cat'], inplace = True)

    # Coerce the new roast_cat column into a numeric type.
    df_coffee['roast_cat'] = df_coffee['roast_cat'].astype(int)

    return df_coffee

def fit_decision_tree():
    '''
    Fits a decision tree regressor with 100g_USD and roast_cat
    as the predictors and rating as the response.

    Parameters:
    No input values.

    Return value:
    No return value but it does produce a pickle file
    containing the trained model.
    '''

    # Define the data frame as our newly recoded data set.
    df_coffee = roast_category()

    # Do another 80/20 train and test split.
    df_train, df_test = train_test_split(df_coffee, test_size = 0.2, random_state = 42)

    # Define the features as our pair of predictors.
    features = ['100g_USD', 'roast_cat']

    # Only the two features in the train data set.
    X = df_train[features]

    # Only the response in the train data set.
    y = df_train['rating']

    # Initiate a DecisionTreeRegressor object.
    dt = DecisionTreeRegressor(random_state = 42)
    
    # Fit the model using the features and response.
    dt.fit(X, y)

    # Create a pickle file with the model.
    try:
        with open('model_2.pickle', 'wb') as file:
            pickle.dump(dt, file)
    except Exception as e:
        print(f'An error occured')

if __name__ == "__main__":

    fit_linear_regression()

    fit_decision_tree()


