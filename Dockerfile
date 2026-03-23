FROM mambaorg/micromamba:1.5.8

WORKDIR /app

ENV P2RANK_VERSION=2.4.2
ENV P2RANK_HOME=/app/tools/p2rank_${P2RANK_VERSION}
ENV PATH=${P2RANK_HOME}:$PATH

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.render.yml /tmp/environment.render.yml
RUN micromamba create -y -n salidock -f /tmp/environment.render.yml && \
    micromamba clean --all --yes

RUN mkdir -p /app/tools && \
    micromamba run -n salidock python -c "import os, tarfile, urllib.request; v=os.environ['P2RANK_VERSION']; url=f'https://github.com/rdk/p2rank/releases/download/{v}/p2rank_{v}.tar.gz'; tgz='/tmp/p2rank.tar.gz'; urllib.request.urlretrieve(url, tgz); tarfile.open(tgz, 'r:gz').extractall('/app/tools')" && \
    chmod +x ${P2RANK_HOME}/prank

COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

EXPOSE 10000

CMD ["sh", "-lc", "micromamba run -n salidock uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]