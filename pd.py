import pandas as pd
from watermark import key

# Create a DataFrame with two tuple columns as the index
df = pd.DataFrame(columns=['index', 'value'])

df.set_index('index', inplace=True)

# Add a row with tuple values for the index and data
df.loc[key((0, 0, 0), (1, 2, 3))] = [42]

# Print the value
print(df)
