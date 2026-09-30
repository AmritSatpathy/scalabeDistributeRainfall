# Spark cluster image for the rainfall benchmark (compute tier E2).
#
# Built on python:3.12-slim (NOT a prebuilt Spark image) so the container's
# Python matches the host venv (3.12) and our modern-syntax src/ runs unchanged
# on both driver and executors. We install the SAME pyspark==3.5.3 wheel as the
# host, which bundles `spark-class` + the Standalone master/worker deploy jars,
# so we launch the cluster daemons directly (no separate Spark tarball).
FROM python:3.12-slim

# Java is required by Spark. Temurin/OpenJDK 17 works with Spark 3.5.
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

ENV PYSPARK_PYTHON=python
ENV PYSPARK_DRIVER_PYTHON=python

# Python deps (match the host). torch is intentionally excluded - the Docker
# tier benchmarks the MLlib models only.
RUN pip install --no-cache-dir \
        pyspark==3.5.3 \
        pandas \
        numpy \
        "setuptools<81"

# Put pyspark's bundled scripts (spark-class, spark-submit, ...) on PATH so the
# compose `command:` can call `spark-class` directly. SPARK_HOME is also set so
# spark-class locates its jars.
ENV SPARK_HOME=/usr/local/lib/python3.12/site-packages/pyspark
ENV PATH="${SPARK_HOME}/bin:${PATH}"

WORKDIR /project
