# References

Collection of articles, papers and techniques considered in this project.

##

- Knowledge Distillation: Teacher is GPT-4. Generate great Go samples on which to fine-tune the model.

## Papers

| Name                                                                                                                                                                                                                                                                    | Description                                                                                    |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [Evaluating Large Language Models Trained on Code](https://www.notion.so/rojasdiego/Evaluating-Large-Language-Models-Trained-on-Code-4ef74247a9ed4f1889449ecca57f76f4?pvs=4)                                                                                            | Foundation Codex paper. HumanEval benchmark and introduction to large language models for Code |
| [CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X](https://www.notion.so/rojasdiego/CodeGeeX-A-Pre-Trained-Model-for-Code-Generation-with-Multilingual-Evaluations-on-HumanEval-X-7bb10770e5d244288112f485152ee439?pvs=4) | 13B multilingual model based on GPT                                                            |

## Articles

| Name                                                                                | Description                                                         |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [How to train your own Large Language Models](https://blog.replit.com/llm-training) | General advice in training code LLMs. Good advice on data pipeline. |

## Libraries

| Name                                                                                                     | Description                                                                                   |
| -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed)                                            | Efficient and fast training of LLMs. Optimized Transformer layers and backprop implementation |
| [NVIDIA FasterTransformer](https://github.com/NVIDIA/FasterTransformer#bert-base-performance-on-pytorch) | Optimized inference of Transformer models                                                     |
| [OpenAI Triton](https://github.com/openai/triton)                                                        | Language and compiler for writing highly efficient custom Deep-Learning primitives on GPUs    |
| [Torch Compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)                  | Compiles PyTorch models into optimised kernels.                                               |
| [PyTorch 2.0 Nightly Release](https://pytorch.org)                                                       | Includes Triton support when using Torch Compile                                              |
