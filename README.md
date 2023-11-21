# Deep Learning Classifier Library

## Purpose

Fine tune pre-trained Deep Learning (DL) model on new dataset

## Setup

Sign up with computing cluster, get cluster access

- For this I'm using [Nautilus Cluster](https://portal.nrp-nautilus.io)

Build docker image and push to dockerhub

- Build on MacOS (Silicon): `docker build --platform linux/x86_64 --rm -t classifier .`
- Build on Windows / Linux / MacOS (Intel / AMD): `docker build --rm -t classifier .`
- Login to dockerhub: `docker login -u user_name`
- Change the tag `classifier` to dockerhub format `user_name/repository_name:tag_name`
    - `docker tag classifier:latest user_name/repository_name:tag_name`
- Push image to dockerhub `docker push user_name/repository_name:tag_name`

Create kubernetes storage

```
cd classifier/kubernetes/storage
for current_file in *.yaml; do kubectl apply -f $current_file; done
kubectl get pvc
```

## Transfer Data

### Option: Kubectl Data Transfer

Create a pod to receive data

```
cd classifier/kubernetes/storage
kubectl apply -f pod.yaml
kubectl get pods
```

Transfer data as compressed file to the pod

`kubectl cp some_data.zip pod_name:path_to_data_storage`

Validate transfer inside of pod

```
kubectl exec -it pod_name -- /bin/bash
cd path_to_data_storage
unzip some_data.zip
```

### S3 Storage

Coming soon!

## Run DL Training Job

Be sure to change parameters inside of `configs/params.yaml`: 
- `data["num_classes"]`: the number of dataset classes
- `paths["test"], `paths["valid"]`, `paths["train"]`: dataset paths

Run a GPU enabled training job with parameters (look & change appropriately) defined at:
- `configs/params.yaml`: general user configuration file
- `classifier/kubernetes/examples/job.yaml`: general kubernetes file

```
cd classifier/kubernetes/examples
kubectl apply -f job.yaml
```
