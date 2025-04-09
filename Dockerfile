FROM python:3.9-slim

RUN apt-get update && apt-get install -y git build-essential

WORKDIR '/i5e2/app'

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./code/app.py .
COPY ./model ./model

EXPOSE 8000

CMD ["python", "app.py"]
