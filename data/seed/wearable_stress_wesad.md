# Wearable Stress Detection and the WESAD Dataset

WESAD (Wearable Stress and Affect Detection) is a publicly available multimodal dataset for studying
stress and affect from physiological signals. It contains recordings from chest- and wrist-worn sensors,
including electrodermal activity (EDA, also called galvanic skin response or GSR), heart rate and blood
volume pulse, skin temperature, respiration, and accelerometer data, collected while participants
experienced baseline, stress, and amusement conditions.

Stress-detection pipelines segment these signals into windows and extract engineered physiological
features, such as heart-rate-variability statistics, EDA phasic and tonic components, and temperature
trends, before training classifiers. Gradient-boosted trees such as LightGBM and XGBoost are popular for
this tabular feature setting because they are accurate and efficient, though deep models are also used.

A key challenge is generalization to unseen users, because physiological baselines vary between
individuals; leave-one-subject-out evaluation gives a realistic estimate of real-world accuracy.
Real-time systems stream sensor data from wearable microcontrollers such as the ESP32, compute features
on the fly, and surface actionable stress insights through a dashboard, enabling continuous, non-invasive
monitoring.
