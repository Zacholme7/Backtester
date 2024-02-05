# Backtester
This is a work in progrss backtester for [this](https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/) indicator. I have re-written it in rust and am now working on developing parameter optimization. No alpha here, this is just for fun and learning

# How does it work?
While the strategy and implementation is pretty complex, the flow is simple. The first step is to edit the config.yml to adjust the backtester/optimizer for the desired configurations. From there, you simply run process.py which will fetch the data, run the backtester, and then generate a plot with all the signals. This is not complete and still has a few minor problems that I am working through, but the core is mainly complete. 
