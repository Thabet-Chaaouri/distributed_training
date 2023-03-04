# distributed_training

### Main concepts:

* Model Parallelism : dividing the model among multiple GPUs (fit large model in different GPUs, but GPU have to wait for each other)
* Data Parallelism, DDP & DP: dividing data for a parallel processing   (GPUs can work in parallel, but each GPU requires a full copy of the model)
* Tensor Parallelism : a from of model parallelism where weights/inputs are spread across GPUs (GPU's work in parallel and avoid memory limitation of copying models, but high communication to synchronize and slow process)
* Pipeline parallelism, Gpipe : combine model and data P (it reduces the bubble, idle times for GPU by processing microbatches in Parallel). none of the transfomrers support PP
* ZeroDP : TP + DP

### Some implemenetations:
* Deepspeed (main feature: ZeRO stage 1, 2 and 3) : integrated with transformers and accelerate from HF
* parallelformers (implements TP,and integrated only for inference with Transformers)
* Megatron LM (Implements TP)
* Varuna
* Fairscale
* SageMaker (can be used with transformers through HF DLC containers)...


### A decision tree for distribution techniques:

Try to fit a model into single memory GPU:
* reduce batch size
* gradient accumulation
* gradient checkpointing
* try 8bit optimizer (to check)

if it does fit, improve speed using:
* DataLoader
* Mixed-precision
* graph compilation
* then try to distibute:
  - multi-GPU simple data parallelism (with accelerate for example)

if it does not fit:
* One single GPU:
  - Deepspeed Zero + CPU offload or NVMe offload + Enable memory centric tiling (if largest layer doesnot fit into single GPU)
* ditributed one node :
	- Zero
* distributed, multi node:
	- Zero


# To do
Try Multi-GPU with accelerate (main process,...)

Try with a TPU (model out the launcher function, use bf16)

Find_executable_batch_size from TOMA 
Gradient accumulation
Gradient Checkpointing
8bit optimizer
Mixed precision

Deepspeed (To read, understand and implement)
FSDP : Fully Sharded Data Parallelism (read, understand and implement model and data parallelism)
MoE

