PROJECT = ofp

WORKSPACE = /home/workspace/Occ_Flow_Pred
DOCKER_IMAGE = x64/ofp:latest
DOCKERFILE ?= Dockerfile

# Change these paths:
dataset_dir=/datasets/Waymo_Motion/
workspace_dir=$(PWD)

DOCKER_OPTS = \
	-it \
	-d \
	--rm \
	-e DISPLAY=${DISPLAY} \
	--gpus '"device= 0"' \
	-e "NVIDIA_DRIVER_CAPABILITIES=all" \
	-e "QT_X11_NO_MITSHM=1" \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	--shm-size=40G \
	--ipc=host \
	--network=host \
	-v $(workspace_dir):$(WORKSPACE):rw \
    -v $(dataset_dir):$(WORKSPACE)/data/Waymo_Motion:rw 

DOCKER_BUILD_ARGS = \
	--build-arg WORKSPACE=$(WORKSPACE) 

COMMAND = python3 ./tools/train.py \
				--gpus 1

NGPUS ?= $(shell nvidia-smi -L | wc -l)
MASTER_ADDR ?= 127.0.0.1
MPI_HOSTS ?= localhost:${NGPUS}
MPI_CMD=mpirun \
		-x LD_LIBRARY_PATH \
		-x PYTHONPATH \
		-x MASTER_ADDR=${MASTER_ADDR} \
		-x NCCL_LL_THRESHOLD=0 \
		-x AWS_ACCESS_KEY_ID \
		-x AWS_SECRET_ACCESS_KEY \
		-x WANDB_ENTITY \
		-x WANDB_API_KEY \
		-np ${NGPUS} \
		-H ${MPI_HOSTS} \
		-x NCCL_SOCKET_IFNAME=^docker0,lo \
		--mca btl_tcp_if_exclude docker0,lo \
		-mca plm_rsh_args 'p 12345' \
		--allow-run-as-root

build:
	docker build \
	$(DOCKER_BUILD_ARGS) \
	-f ./docker/$(DOCKERFILE) \
	-t $(DOCKER_IMAGE) .

start:
	docker run --name $(PROJECT) \
	$(DOCKER_OPTS) \
	$(DOCKER_IMAGE)

into:
	docker exec -it $(PROJECT) /bin/bash -c \
    "cd $(WORKSPACE); nvidia-smi;/bin/bash"

dist-run:
	docker run --name $(PROJECT) --rm \
		-e DISPLAY=${DISPLAY} \
		-v ~/.torch:/root/.torch \
		${DOCKER_OPTS} \
		-v $(PWD):$(WORKSPACE) \
		${DOCKER_IMAGE} \
		${COMMAND}

docker-run: 
	docker exec -it $(PROJECT) /bin/bash -c \
    "cd $(WORKSPACE); ${COMMAND};"
		
docker-run-mpi: docker-build
	docker run ${DOCKER_OPTS} -v $(PWD)/outputs:$(WORKSPACE)/outputs ${DOCKER_IMAGE} \
		bash -c "${MPI_CMD} ${COMMAND}"

clean:
	find . -name '"*.pyc' | xargs rm -f && \
	find . -name '__pycache__' | xargs rm -rf

stop:
	docker stop $(PROJECT)