# Stock Trading with Deep Learning: When GameStop meets LSTMs
Source code for paper: https://arxiv.org/abs/2103.01121

- [x] S&P 500 Buy & Hold - Yigit
- [x] LSTM Prediction - Yigit
- [x] Consolidation Break out - Yigit

## Development
- pip install -r requirements.txt (Do this once)
- python main.py


## Current Results

### S&P start_date=2002-02-13, end_date=2021-02-12
```
[INFO] Pipeline started for Ticker:SPY Start date:2002-02-13 End date:2021-02-12
[INFO] Buy and Hold profit is $281.1600112915039
[INFO] Pure LSTM-based trading profit is $110.29022216796918 Profitable trade count: 462 Unprofitable trade count: 401 Success Rate: 53.53418308227115 Test dataset final RMSE value:8.958006168683683
[INFO] Consolidation Breakout trading profit is $4.347592096226596 Profitable trade count: 536 Unprofitable trade count: 395 Success Rate: 57.57250268528465%
```

### GME start_date=2002-02-13, end_date=2021-02-12
```
[INFO] Pipeline started for Ticker:GME Start date:2002-02-13 End date:2021-02-12
[INFO] Buy and Hold profit is $45.63333559036254
[INFO] Pure LSTM-based trading profit is $69.04612421989441 Profitable trade count: 213 Unprofitable trade count: 239 Success Rate: 47.123893805309734% Test dataset final RMSE value:13.458590932218822
[INFO] Consolidation Breakout trading profit is $2.1027738249293195 Profitable trade count: 156 Unprofitable trade count: 169 Success Rate: 48.0%
```

Please include the following citation if you cite this paper:
```
@misc{alparslan2021extreme,
      title={Extreme Volatility Prediction in Stock Market: When GameStop meets Long Short-Term Memory Networks}, 
      author={Yigit Alparslan and Edward Kim},
      year={2021},
      eprint={2103.01121},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
