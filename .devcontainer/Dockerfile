FROM mcr.microsoft.com/devcontainers/python:3.10

COPY zscaler-bundle.pem /tmp/zscaler.pem
RUN if [ -s /tmp/zscaler.pem ]; then \
      cp /tmp/zscaler.pem /usr/local/share/ca-certificates/zscaler.crt && \
      update-ca-certificates; \
    fi && rm -f /tmp/zscaler.pem

RUN find /etc/apt /usr/share/keyrings -name "*yarn*" -delete 2>/dev/null || true
