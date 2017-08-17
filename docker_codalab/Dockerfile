FROM ubuntu

RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  python-pip \
  swig \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

#install slearn
RUN pip install numpy==1.12.0
RUN pip install scipy==0.16.0
RUN pip install scikit-learn==0.16.1 

#install auto-sklearn dependencies
RUN pip install jupyter
RUN pip install psutil==3.2.2
RUN pip install pyyaml==3.11
RUN pip install pyparsing==2.1.10
RUN pip install pandas==0.17.0



