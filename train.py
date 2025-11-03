import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pickle

def get_coffee_data():

    df_coffee = pd.read_csv("https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv")

    df_coffee.drop_duplicates(subset = 'desc_1', inplace = True)

    df_coffee.dropna(subset = ['desc_1', 'roast', 'loc_country'], inplace = True)

    return df_coffee

def fit_linear_regression():

    df_coffee = get_coffee_data()

    df_train, df_test = train_test_split(df_coffee, test_size = 0.2, random_state = 42)

    features = ['100g_USD']

    X = df_train[features]

    y = df_train['rating']

    lm = LinearRegression()

    lm.fit(X, y)

    try:
        with open('model_1.pickle', 'wb') as file:
            pickle.dump(lm, file)
    except Exception as e:
        print(f'An error occured')


def roast_category():
    
    df_coffee = get_coffee_data()

    roast_map = {'Light' : 1, 'Medium-Light' : 2, 'Medium' : 3, 'Dark' : 4}

    df_coffee['roast_cat'] = df_coffee['roast'].map(color_map)

    return df_coffee

def fit_decision_tree():

    df_coffee = roast_category()

    df_train, df_test = train_test_split(df_coffee, test_size = 0.2, random_state = 42)

    features = ['100g_USD', 'roast_cat']

    X = df_train[features]

    y = df_train['rating']

    dt = DecisionTreeRegressor(random_state = 42)
    
    dt.fit(X, y)

    try:
        with open('model_2.pickle', 'wb') as file:
            pickle.dump(dt, file)
    except Exception as e:
        print(f'An error occured')

if __name__ == "__main__":

    fit_linear_regression()

    fit_decision_tree()


