| Model | Aggregated Model? | parameter                        | Extra data | train MAE | test MAE |
|-------|-------------------|----------------------------------|------------|-----------|----------|
| ARIMA |                   | p=5,d=1,q=0                      |            | N/A       | 0.6405   |
| LSTM  |                   | One hidden layer with 16 neurons |            | ???       | 1.0343   |
| LSTM  | Yes               | One hidden layer with 16 neurons |            | ???       | 0.789    |
| LSTM  |                   | One hidden layer with 16 neurons | Neighbor   | ???       | ???      |
| LSTM  | Yes               | One hidden layer with 16 neurons | Neighbor   | 0.9446    | 0.909    |
| LSTM  |                   | One hidden layer with 16 neurons | Twitter    | ???       | ???      |
| LSTM  | Yes               | One hidden layer with 16 neurons | Twitter    | ???       | ???      |