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

## Run DL Training Job:

