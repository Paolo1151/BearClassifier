FROM python:3.9

RUN mkdir /dash

COPY requirements.txt /dash

WORKDIR /dash

RUN pip install -r requirements.txt

COPY . /dash

EXPOSE 8501

CMD ["streamlit", "run", "dash.py"]