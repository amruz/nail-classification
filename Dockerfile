FROM python:3.5
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r ./requirements.txt
COPY nail-it.py /app
COPY nails_fcl_vgg16.h5 /app
ENV KERAS_BACKEND=theano
CMD ["python", "nail-it.py"]~
