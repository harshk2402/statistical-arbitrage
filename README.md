Retrieve all the data using Bloomberg, Quandl, Yahoo Finance, Google Finance and WRDS as discussed in class. I will provide guidance on how to have access to WRDS
Volatility signature:
using the standard approach (squared root of variance of log returns adjusted for the sampling frequency: we assume 252 trading days per year) to compute volatility

     1-1) Download intra-day and daily data for foreign currencies, S&P, DJA futures or ETFs  data using  either R, Python, C++, Java etc..
     1-2) Compute the annualized volatility graph  with a sampling frequency from 1 day returns to 30 days returns
     1-2-a) Produce the graph at different time periods i.e. 2000 to 2007, and 2007 to 2014

     1-2-b) compute the mean, the median, the 25% and 75% quantile for the rolling volatility signature (i.e 6months or 1 year window) for each frequency and plot the corresponding graphs
      1-3) compute the volatility signature using intraday with annualized returns. the sampling frequency could range from seconds or minutes to few hours.
      1-3-a) compute the mean, the median,  the 25% and 75% quantile for the rolling volatility signature (i.e 6months or 1 year window) for each frequency and plot the corresponding graphs
