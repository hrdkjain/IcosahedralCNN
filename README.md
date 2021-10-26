This project contains PyTorch implementation of Cohen, Taco, et al. "Gauge Equivariant Convolutional Networks and the Icosahedral CNN." with an example. Optionally the project can be run using docker with the provided `dockerfile`

## Usage
The project has been implemented on python and can be run using `Dockerfile`
 
#### 1. Build Docker Deploy Image
```yaml
docker build --network=host \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER=docker_$USER \
  -t githubicosahedralcnndocker:v1 . 
```
#### 2. Run Docker Image
```yaml
docker run --network host \
  --gpus all -it --rm --user $(id -u):$(id -g) \
  --shm-size=8G \
  -e HOST_HOSTNAME=`hostname` \
  --mount type=bind,src=<>,dst=<> \
  --workdir <> githubicosahedralcnndocker:v1 bash
```
#### 3. Training Example
```yaml
python3 examples/mnist/run.py 
```

###  Projects
If you like this project you also might be interested in other projects which use IcosahedralCNN as remeshing stage for [mesh generation](https://github.com/hrdkjain/GenIcoNet).