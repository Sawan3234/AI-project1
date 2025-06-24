import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 


df = pd.read_excel("homeprice.xlsx")
print(df.columns)


plt.xlabel('area (sq ft)')
plt.ylabel('prices (NPR)')
plt.scatter(df['area'], df['prices'], color='red', marker='+')


reg = linear_model.LinearRegression()
reg.fit(df[['area']], df['prices'])


predicted_prices = reg.predict([[3300]])
print(f"Predicted price for 3300 sq ft area: {predicted_prices[0]}")


plt.plot(df['area'], reg.predict(df[['area']]), color='blue')
plt.show()
coef=reg.coef_
print(coef)
#predicting the prices
d=pd.read_excel("area.xlsx")
d.head(3)
p=reg.predict(d)
d['prices']=p
print(p)
d.to_xlsx("prdeiction.xlsx",index=False)
