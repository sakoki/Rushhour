| Model | Aggregated Model? | parameter                                                 | Extra data | train MAE | test MAE |
|-------|-------------------|-----------------------------------------------------------|------------|-----------|----------|
| ARIMA |                   | p=5,d=1,q=0                                               |            | N/A       | 0.6405   |
| LSTM  |                   | One hidden layer with 16 neurons, 8 epoch, time window=12 |            | 1.1313    | 1.0343   |
| LSTM  | Yes               |                                                           |            | 0.9026    | 0.789    |
| LSTM  |                   |                                                           | Neighbor   | 1.1325    | 1.0355   |
| LSTM  | Yes               |                                                           | Neighbor   | 0.9446    | 0.909    |
| LSTM  |                   |                                                           | Twitter    | ???       | ???      |
| LSTM  | Yes               |                                                           | Twitter    | ???       | ???      |
| LSTM  |                   |                                                           | 5 month    |           | 1.104    |
| ARIMA |                   |                                                           | 5 month    | N/A       | 0.8428   |
