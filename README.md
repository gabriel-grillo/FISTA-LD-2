To build the image run
```
docker build --build-arg COMP_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.') -t fista-ld .
```

To run the container run
```
docker run -it --rm --runtime=nvidia --gpus '"device=0"' --ipc=host -p 8888:8888 -v $(pwd)/code:/code fista-ld
```
