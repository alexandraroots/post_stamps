FROM python:3.8

RUN apt update

RUN apt update && apt install -y libgl1-mesa-glx libglib2.0-0

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

COPY . /app

WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python3.8", "main.py"]
