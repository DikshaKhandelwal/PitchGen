import pandas as pd
df = pd.read_csv("all-data.csv", encoding='ISO-8859-1', header=None)
print(df.head(3))