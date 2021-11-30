## crypto000 - A crypto trading bot

### Some Stats

- Peak ROI ~5.5% 
- Can trade more than 100 cryptos simultaneously


### Requirements 

ccxt 
numpy
Flask
numba


### How to use


- `python crypto000.py`

- Flask will be running in the background, serving a small API


### How it works & Motivation


- Calculate two exponential moving averages and if they cross it is either time to sell or buy depending on how the lines crossed 
- This is all done in threads with Queues, theoretically if there were no rate limits by the exchange APIs, you could trade on nearly infinity many markets


The creator of this script knows that this provides absolutely 0 value to anyone besides the user, I have started and abandoned similar projects multiple times 
because I know that at the end no real value is created, my time would be better spent doing other things and exploring deeper problems. But the time I already spent
on this project made me want to finish it so here I am. You could write 5 lines of code and this would be live and trading on real money. I will probably do that 
some time but I also know that money is only temporary and will never buy me the happiness I want. This will be the last project by me that is only focussed on the money.


At the end of the day my hope is maybe someone reads this script and it is helpful to them. What a waste of time and energy this was. 


Have fun with it tho. 


### TODOs 


- Additional trading strategies (Maybe do some fancy NLP?)




