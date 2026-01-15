# Stage 1: Builder stage
FROM python:3.12.12-slim-bookworm as builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget ca-certificates curl gcc python3-dev openjdk-17-jdk-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Definir versión de Spark 4.0.1
ENV SPARK_VERSION=4.0.1
ENV HADOOP_VERSION=3
ENV SPARK_FILENAME="spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"

RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/${SPARK_FILENAME} -O ${SPARK_FILENAME} \
    && tar xzf ${SPARK_FILENAME} -C /tmp && rm ${SPARK_FILENAME} \
    && mv /tmp/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /spark

# --- JARs específicos solicitados ---
# Hadoop AWS 3.4.2
RUN wget -O /spark/jars/hadoop-aws-3.4.2.jar \
    https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.4.2/hadoop-aws-3.4.2.jar

# AWS SDK Bundle 1.12.793 (Versión bundle para evitar conflictos en Spark)
RUN wget -O /spark/jars/aws-java-sdk-bundle-1.12.793.jar \
    https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.793/aws-java-sdk-bundle-1.12.793.jar

# Entorno virtual y dependencias de Python
RUN mkdir /app && python -m venv /opt/venv
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install uv==0.9.24 \
    && uv pip install --system -r requirements.txt

# Copiamos todo el proyecto (main.py, utils.py, preprocessing_001.py, etc.)
COPY . .

# Stage 2: Runtime stage
FROM python:3.12.12-slim-bookworm as runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless procps ca-certificates curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar artefactos
COPY --from=builder /spark /spark
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Configuración de entorno
ENV SPARK_HOME=/spark \
    PATH="/spark/bin:/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.9.9-src.zip" \
    PYTHONUNBUFFERED=1

# Configuración de logs a nivel ERROR para limpiar la consola
COPY --from=builder /spark/conf/log4j2.properties.template /spark/conf/log4j2.properties
RUN sed -i 's/rootLogger.level = info/rootLogger.level = error/' /spark/conf/log4j2.properties

# Usuario de ejecución
RUN useradd -m sparkuser && chown -R sparkuser:sparkuser /app /spark
USER sparkuser

