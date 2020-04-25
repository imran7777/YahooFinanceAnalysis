#!/usr/bin/env python
# coding: utf-8

# # Task by Blinkist
# ## Find the best time to buy and sell a "product" to maximize the profit depending on the historical data.

# ### Probelm Statement:
# #### We have a given data set for a stock from past year. We would like to find out the best day to buy and sell that stock to be able to earn the maximum amount of profit. We are suppose to buy and sell only once.
# #### Testing our solution with an investment amount of 100k USD to varify our solution.

# We have used python libraries pandas, numpy, matplotlib, seaborn and statsmodels

# In[1]:


# Importing relevant libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ### Loading a file with the historical data

# In[2]:


# Asking user to give the file name to be loaded
data = pd.read_csv(input('Name of file to be analysed (must be .csv file) : '))


# In[3]:


data


# In[4]:


# We can automate or make it user defined to be able to use it for any dataset
#data = pd.read_csv('BTC-USD.csv')
#data


# ### # Go through the data to detect anomalies

# In[5]:


data.describe()


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data['High'].plot()


# In[9]:


data['Low'].plot()


# ### Visualizing data to have a better understanding of the data and its features 

# In[10]:


# Plotting graph of Highest and Lowest values during the given time frame
data[['High', 'Low']].plot()
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("BTC-USD Price data")
plt.show()


# In[11]:


# Plotting graph of Opening price and Closing price values during the given time frame
data[['Open', 'Close']].plot()
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("BTC-USD Price data")
plt.show()


# ## Feature consideration

# #### Going through the data and researching about the important features involved in a statistical data set we found out the best price indicator is "Adj Close" column.
# ### Adj Close:
# #### What is an Adjusted Closing Price?
#     Adjusted closing price amends a stock's closing price to accurately reflect that stock's value after accounting for any corporate actions. It is considered to be the true price of that stock and is often used when examining historical returns or performing a detailed analysis of historical returns.
# 
# #### KEY TAKEAWAYS:
# 1 - Adjusted closing price amends a stock's closing price to accurately reflect that stock's value after accounting for any corporate actions.
# 
# 2 - The closing price is the 'raw' price which is just the cash value of the last transacted price before the market closes.
# 
# 3 - Adjusted closing price factors in corporate actions such as stock splits, dividends / distributions and rights offerings.

# In[12]:


# Plotting "Adj Close" price against time to visualize the variation in data sample.
data['Adj Close'].plot()
plt.xlabel("Date")
plt.ylabel("Adjusted Close")
plt.title("BTC-USD Price data")
plt.show()


# In[13]:


# Before performing any manipulation over a dataset generating a copy (milestone) is widely highly recommended practice
df = data.copy()
df


# #### Setting up Date as index
# 

# In[14]:


# We set up "Date" as index because we have to track the time to buy/sell a product on the bases of Time.
df.set_index("Date", inplace=True)


# In[15]:


# To see if Date is set as index
df


# In[16]:


df['High'].plot()


# In[17]:


df['Low'].plot()


# In[18]:


# Generated a copy incase we need it later for further analyses
df_1= df.copy()
df_1


# In[19]:


df_1[['Low','Volume']].plot()


# In[20]:


df_1 [['Adj Close', 'Volume']].plot()


# ### Dropping columns
# In the next step we drop the unnecessary columns because from our research we found out that "Adj Close" is the feature we will consider for best selling and buying time.

# In[21]:


df.drop(['Open', 'Close', 'Low', 'High', 'Volume'], axis=1, inplace=True)


# In[22]:


# Now we have only Adj Close column indexed by dates
df


# In[23]:


# Plotting Adj Close against time
df.plot()


# In[24]:


df['Adj Close'].plot()
plt.xlabel("Date")
plt.ylabel("Adjusted Close")
plt.title("BTC-USD Price data")
plt.show()


# In[25]:


# Check the type of data to be able to know which methods can be performed
type(df)


# #### DataFrame to Numpy conversion

# In[26]:


# converting Dataframe to numpy array
x = df.to_numpy()
x


# ### Now we have an array ready to be analysed for the best time to BUY / Sell

# To be able to find the best buying and selling price we need to create a function that takes Adj Close values and finds out the two values (numbers) that have the maximum difference between them but considering that the larger value appears after the smaller value. (buying must be cheaper than selling price to be able to earn the maximum amount of profit.
# P.S If Larger vlaue == Smaller value this function will return 0. (i.e 4300 == 4300)

# ### Function to get the best buy / sell prices

# In[27]:


def maxProfit(arr, arr_size): 
    max_diff = arr[1] - arr[0] 
    max_value= 0.0
    min_value= 0.0
    for i in range( 0, arr_size ): 
        for j in range( i+1, arr_size ): 
            if(arr[j] - arr[i] > max_diff):  
                max_diff = float(arr[j] - arr[i])
                min_value= float(arr[i])
                max_value= float(arr[j])
    result = [max_diff , min_value, max_value]
    return result


# #### Time Complexity : O(n^2)
# #### Auxiliary Space : O(1)

# In[28]:


# Assigning parameter vlaues
arr = x
size = len(x)


# In[29]:


# Calling the function in a variable to store the resulting values.
res = maxProfit(arr, size)
res


# In the above list 1st vlaue ([0]) is the profit earned, 2nd value ([1]) is buying price and 3rd value ([1]) is selling price

# In[30]:


# We can print values by there location in the list
print ("Maximum Profit is", res[0])
print ("Best buying price of the stock in whole year =", res[1])
print ("Best selling price of the stock in whole year =", res[2])


# ## Best date to buy and sell BTC in last year
# To be able to extract dates related to these values we need to iterate through the column 'Adj Close' in dataframe "df".
# What happens is these values stored in the list 'res' will be compared in the column 'Adj Close' and give us back the dates related to those values.

# In[31]:


data.head(6)


# In[32]:


# Looking for the best buying price date in dataframe 'data'
print ("Best time to buy stock in last year US$ :", data[data['Adj Close']== res[1]])


# The above result shows that the best date to buy was march 25, 2019 where Openening price was "4024.112793 USD", Highest price was "4038.840820 USD", Lowest Price was "3934.031250 USD", Closing Price was "3963.070557", Adjusted Price was "3963.070557" and Volume of shares was "10359818882"

# In[33]:


# Looking for the best selling price date in dataframe 'df'
print ("Best time to sell stock in last year US$ :", df[df['Adj Close']== res[2]])


# In[34]:


# Maximum profit earned by the transaction
print ("Maximum Profit earned in last year in US$ :", res[0])


# ### Lets see how much could have been earned with 100k capital.

# In[35]:


capital = float(input('Enter amount in USD: '))
# To be able to calculate profit for capital amount we will 
# total_profit = number of shares bought on minimum price * maximum_price of a share
total_profit = float((capital / res[1]) * res[2])
print ('total profit earned on your investment in USD:',round(total_profit,4))


# ### Limitations of design and Future work to improve profits

# As we have used the historical data and picked the best buying and selling prices to maximize the profit which is not considered to be the best solution for future investments.
# We can use ML / DL (Machine Learning / Deep Leaning) libaries to run regression analyses on the data to predict the best possible buying / selling prices of a stock for the future investment plans. 
# These machine learning algorythms allow us to consider multiple features of a dataset (not just Adj Close prices) and can immensly increase our profits.
# 
# Moreover the profits can be increased by splitting the investment into multiple chunks for different quarters of a year.
# The input from the finance department is a key in any investment as finance professionals. They can bring in their experience from their previous understanding of the market trends.
# 
# 
# 
# 

# In[ ]:




