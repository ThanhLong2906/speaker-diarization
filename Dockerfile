FROM nguyenlong2906/fastapi-base:python3.10-slim AS huh

WORKDIR /app

# COPY ./requirements.txt ./

# #COPY ./ /code/

# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install -r requirements.txt

    # pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu 
    
# RUN apt-get install g++ gcc -y && \
#     pip install youtokentome
COPY ./src /app

# # install ffmpeg
# RUN apt-get update && \
#     apt-get install -y \
#     libsndfile1 \
#     ffmpeg && \
#     rm -rf /var/lib/apt/lists/*

# FROM huh as dev-envs

# RUN <<EOF
# apt-get update
# apt-get install -y --no-install-recommends git
# EOF

# RUN <<EOF
# useradd -s /bin/bash -m vscode
# groupadd docker
# usermod -aG docker vscode
# EOF


# # install Docker tools (cli, buildx, compose)
# COPY --from=gloursdocker/docker / /

#RUN chmod +x /code/start_docker

#CMD /code/start_docker
