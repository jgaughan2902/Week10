import pandas as pd

def get_coffee_data():
    df_coffee = pd.read_csv("https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv")

    df_coffee.drop_duplicates(subset = 'desc_1', inplace = True)

    df_coffee.dropna(subset = ['desc_1', 'roast', 'loc_country'], inplace = True)

    return df_coffee