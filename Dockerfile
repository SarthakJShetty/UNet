FROM python:3.11

WORKDIR /opt/

COPY . /opt/

RUN pip install --upgrade pip

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install -e .

CMD [ "python", "scripts/train.py"]
