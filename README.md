### Things to contemplate

- Allow/Disallow selling with losses, because prioritizing strategy over short term profit will benefit in the long term? 

- Commission fee (~0.1%) to broker, less trades means less commission but also more volatility because you rely on less trades to make profits -> find good balance 

- In order to not exceed the rate limit implementing some local caching or database would be best 


#### Selling with losses

This is ofc not good but holding a dying asset is also not good because your money is bound and cannot be spent in other markets 


### Steps 

~~ 1. for every symbol pair: Past data -> generate b, e pair ~~

~~ 2. setup live actor that trades on new data ~~

3. serve api

4. rich rich 


### Potential Improvements 

~~ - Local OHLC Data caching / Database ~~ 
- Multithreaded learning 