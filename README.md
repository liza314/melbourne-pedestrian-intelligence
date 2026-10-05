
# Melbourne Pedestrian Intelligence

Interactive Melbourne pedestrian analytics and AI forecasting dashboard.

## Features
- Multi-location pedestrian forecasts
- TimesFM 3 foundation model
- 7-day hourly forecasting
- 80% prediction intervals
- Model evaluation against a weekly baseline
- Unusual-activity detection
- Interactive Streamlit dashboard

## Final Holdout Performance
- TimesFM WAPE: 8.49%
- Weekly baseline WAPE: 11.69%
- WAPE improvement: 27.4%
- 456 unseen test hours

## Run
pip install -r requirements.txt
streamlit run app.py

Pedestrian counts are sensor detections, not unique people.
