FROM apache/airflow:2.9.1

USER airflow

ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow"

RUN pip install --no-cache-dir \
    scikit-learn>=1.4.0 \
    pandas>=2.2.0 \
    numpy>=1.26.0 \
    joblib>=1.3.0 \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0