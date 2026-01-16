############################
# Stage 1: Python builder
############################
FROM python:3.11.14-slim-bookworm AS python-builder

WORKDIR /deps

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --prefix=/usr/local -r requirements.txt

############################
# Stage 2: Spark runtime
############################

FROM spark:4.0.1

USER root

# Copiar site-packages y binarios Python
COPY --from=python-builder /usr/local /usr/local

# Instalar Python3 en la imagen runtime y crear `python` -> `python3`
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-distutils wget unzip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && python --version

# -------------------------------
# Aquí agregamos los JARs de S3
# -------------------------------
# Crear carpeta para jars si no existe
RUN mkdir -p /opt/spark/jars

# Descargar hadoop-aws, aws-java-sdk y commons-pool2
RUN wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.4.1/hadoop-aws-3.4.1.jar \
    # && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.797/aws-java-sdk-bundle-1.12.797.jar \
    \
    # Añadidas dependencias SDK aws
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/s3/2.41.8/s3-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/sdk-core/2.41.8/sdk-core-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/annotations/2.41.8/annotations-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/utils/2.41.8/utils-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/aws-core/2.41.8/aws-core-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/auth/2.41.8/auth-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/regions/2.41.8/regions-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/http-client-spi/2.41.8/http-client-spi-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/netty-nio-client/2.41.8/netty-nio-client-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/protocol-core/2.41.8/protocol-core-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/profiles/2.41.8/profiles-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/endpoints-spi/2.41.8/endpoints-spi-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/identity-spi/2.41.8/identity-spi-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/s3-transfer-manager/2.41.8/s3-transfer-manager-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/metrics-spi/2.41.8/metrics-spi-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/apache-client/2.41.8/apache-client-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/retries-spi/2.41.8/retries-spi-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/retries/2.41.8/retries-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/http-auth-spi/2.41.8/http-auth-spi-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/http-auth/2.41.8/http-auth-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/http-auth-aws/2.41.8/http-auth-aws-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/org/reactivestreams/reactive-streams/1.0.4/reactive-streams-1.0.4.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/aws-xml-protocol/2.41.8/aws-xml-protocol-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/aws-query-protocol/2.41.8/aws-query-protocol-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/checksums-spi/2.41.8/checksums-spi-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/checksums/2.41.8/checksums-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/aws-json-protocol/2.41.8/aws-json-protocol-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/aws-cbor-protocol/2.41.8/aws-cbor-protocol-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/smithy-rpcv2-protocol/2.41.8/smithy-rpcv2-protocol-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/json-utils/2.41.8/json-utils-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/third-party-jackson-core/2.41.8/third-party-jackson-core-2.41.8.jar
    
USER spark