# from python
FROM python:3.13

# install mlflow
RUN pip install --no-cache-dir mlflow==3.5.1 

# start mlflow on port 5000
EXPOSE 5000
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]