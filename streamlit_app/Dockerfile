FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY streamlit_app/requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY streamlit_app/app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]