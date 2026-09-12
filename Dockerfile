FROM python:3.11-slim

RUN pip install --no-cache-dir uv

# コンテナ自体が隔離環境のため.venvは作らず、システムPythonに直接インストールする
ENV UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /workspace

# 依存関係を先に入れる。コードだけ変えた際に再インストールが走らないようにするため
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 本番用にコードをイメージへ焼き込む。
# 開発（Devコンテナ）ではホストの . が /workspace にマウントされて上書きされるため影響しない
COPY src/ ./src/
COPY ui/ ./ui/
COPY .streamlit/ ./.streamlit/
