FROM rayproject/ray:2.52.0-py311

# 1️⃣ Instalar uv (binario precompilado, súper rápido)
RUN pip install --no-cache-dir uv

# 2️⃣ Copiar requirements primero (mejor cache de Docker)
COPY requirements.txt /tmp/requirements.txt

# 3️⃣ Instalar deps con uv (mucho más rápido que pip)
RUN uv pip install \
    --system \
    --no-cache \
    -r /tmp/requirements.txt

# Limpiar cachés temporales
#RUN rm -rf /root/.cache \
#           /tmp/* \
#           /var/tmp/*