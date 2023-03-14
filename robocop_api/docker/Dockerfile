FROM python:3.8-slim

COPY ./docker/requirements.txt .

RUN apt-get update && apt-get install -y wget gnupg curl && apt-get install --no-install-recommends -y \
    libgl1 \
    libgomp1 \
    python3-pip


RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install uvicorn==0.20.0
WORKDIR /app/robocop_api

CMD ["python3","main.py"]
