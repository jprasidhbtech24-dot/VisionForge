FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY README.md ./README.md
ENV PYTHONUNBUFFERED=1 PORT=7860
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
