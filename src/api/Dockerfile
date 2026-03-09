FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv pip install --system .

COPY . .
EXPOSE 80
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "80"]