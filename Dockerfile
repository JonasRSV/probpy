FROM python

RUN mkdir probpy_demo
WORKDIR probpy_demo

COPY probpy probpy
ADD setup.py .

RUN python3 setup.py install
RUN pip3 install jupyter
RUN pip3 install matplotlib==3.1.0
RUN pip3 install seaborn
