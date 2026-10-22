#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define paths:
ifeq ($(origin SRCDIR), undefined)
	SRCDIR := $(shell pwd)/src
endif
ifeq ($(origin TEMPDIR), undefined)
	TEMPDIR := $(shell pwd)/tmp
endif
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.PHONY: all
all: $(TEMPDIR)/cuda_radon.so pythondir $(TEMPDIR)/cuda_radon_torch.so

.PHONY: pythondir
pythondir:
	mkdir -p $(TEMPDIR)
	cp -f $(SRCDIR)/python/*.py $(TEMPDIR)
	ln -sf $(SRCDIR)/python/*.ipynb $(TEMPDIR)

ifndef COMP_CAP
$(error COMP_CAP not defined. Ex: make COMP_CAP=86)
endif
ARCHS := -gencode arch=compute_$(COMP_CAP),code=sm_$(COMP_CAP)
$(TEMPDIR)/cuda_radon.so : $(TEMPDIR)/cuda_radon.o
	cd $(TEMPDIR) && gcc $(TEMPDIR)/cuda_radon.o -shared -o $(TEMPDIR)/cuda_radon.so
$(TEMPDIR)/cuda_radon.o : $(SRCDIR)/c/raft_cuda/cuda_radon.cu
	mkdir -p $(TEMPDIR)
	nvcc $(CCBIN) -c -O3 --compiler-options '-fPIC' -o $(TEMPDIR)/cuda_radon.o -m64 $(ARCHS) $(SRCDIR)/c/raft_cuda/cuda_radon.cu -lstdc++ -lpthread -lm
	cp $(SRCDIR)/python/* $(TEMPDIR)
$(TEMPDIR)/cuda_radon_torch.so : $(TEMPDIR)/cuda_radon_torch.o
	cd $(TEMPDIR) && gcc $(TEMPDIR)/cuda_radon_torch.o -shared -o $(TEMPDIR)/cuda_radon_torch.so
$(TEMPDIR)/cuda_radon_torch.o : $(SRCDIR)/c/raft_cuda/cuda_radon_torch.cu
	mkdir -p $(TEMPDIR)
	nvcc $(CCBIN) -c -O3 --compiler-options '-fPIC' -o $(TEMPDIR)/cuda_radon_torch.o -m64 $(ARCHS) $(SRCDIR)/c/raft_cuda/cuda_radon_torch.cu -lstdc++ -lpthread -lm

