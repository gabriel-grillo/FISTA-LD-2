# Fork de elias-helou/ray_tracing para adaptação para Tensorflow

Para fazer a build da imagem do container, rode:
```
docker build --build-arg COMP_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.') -t ray_tracing .
```

Depois rode o container com:
```
docker run -it --rm --runtime=nvidia --gpus '"device=0"' --ipc=host -p 8888:8888 -v $(pwd)/workspace:/workspace ray_tracing
```

Se quiser abrir um notebook, use o seguinte comando dentro do container:
```
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```