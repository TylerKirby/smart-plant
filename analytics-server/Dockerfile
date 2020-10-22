FROM tiangolo/uvicorn-gunicorn:python3.7

RUN apt-get update
RUN apt-get install python3


WORKDIR /app

COPY requirements.txt /app
RUN pip3 install --no-cache-dir -r requirements.txt

COPY main.py /app

ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0","--port", "5000"]