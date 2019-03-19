
# File Description
_Created on : 2/18/2019 Updated on:2/23/2019_

    |- __pychache__
    |- utils
        |- __init__.py
        |- decorators.py
        |- formatter.py
        |- toolkit.py
        |- generator.py
    |- models
        |- ARIMA_baseline.py
        |- LSTM_baseline.py
    |- playground
        |- crawling_sfmta_data.ipynb
        |- exploratory.ipynb
        |- ARIMA_baseline.ipynb
        |- Census_Tracts_Exploratoy.ipynb
        |- nyc_map.ipynb
        |- prediction_table_generator.ipynb
        |- region_mapper.ipynb
        |- sfmta_data_analysis.ipynb
    |- scraping_sfmta.py
    |- gen_region_files.py

## Explanations of directory structures:
- Files in playground dir are exploratory codes in a format of ipynb(jupyter notebook).
- Files in utils dir contains different kinds of utility functions that can be reused by pipeline code.
- Files in models dir contains models that predict Y given timeseries data(both baseline and improved model) and twitter data (improved model).
- Files directly under code files are pipline files only contains skeleton code needs to call functions in utils and models.


- Choosing time frame of tweet data (window of dates, how much in advance?)
    - Mention an event prior to its occurrence e.g., "Excited for the SF Giants baseball home game on March 17th!"
        - Indication of location: "SF Giants" + "home" --> SF Giants Stadium
        - Indication of event date: March 17th
    -   Mention a event during the onset of its occurrence. e.g., "Horrible traffic accident on Highway 17"
        - Indication of location: "Highway 17"
        - Indication of time: time tweet was written
- Take into account expected traffic times (morning rush hour 7:00 am - 10:00 am)
- Normalize to compare differences rather than exact times?
