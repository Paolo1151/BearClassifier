FROM tiangolo/uvicorn-gunicorn:python3.9

RUN mkdir /bear

COPY requirements.txt /bear

WORKDIR /bear

RUN pip install -r requirements.txt

COPY . /bear

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]