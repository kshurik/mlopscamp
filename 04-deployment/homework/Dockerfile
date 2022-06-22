FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install pipenv

COPY ["./app/Pipfile", "./app/Pipfile.lock", "./app/*.py","./"] 

RUN pipenv install --deploy --system

WORKDIR /app
CMD ["python", "./starter.py"]
