FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /raccoon
COPY . /raccoon
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 80

# matplotlib rendering without display
ENV MPLBACKEND "agg"

CMD ["bash"]
