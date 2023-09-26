*******************************
DL WorkFlow
*******************************
- DL WorkFlow
	- Idea theory
	- Design Experiments
		- Dataloaders and datasets 2
		- Pretrained models (torchvision, torchtext and torchaudio) 2
		- Preprocess data/setting env 2
	- Implements model
		- Interfacing with environements (from soumith video, I suspect is with GYM) 2
			- any env with python api (show some example)
		- Debugging and profiling 3
			- bottleneck
			- profiler
	- Training & validation - splits again part of data loaders
		- Saving checkpoints or intermediate stages 1
		- Ignite 3
	- Publish and ship - covers in pytorch to production
		- Distributed PyTorch 2
		- Checkpointing
		- Loading and dumping models (storage.cude('device'))
		- Compilation and JIT
		- Saving model - equivalant to checkpointing in TF

TODO
K fold
Plotting validation and testing score