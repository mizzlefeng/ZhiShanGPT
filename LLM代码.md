# 创建Conda环境

```shell
conda create -n llm python=3.10
```

# 初步运行

```shell
# 句子补全
bsub -Is -J llama -q normal_test -gpu "num=1:mode=exclusive_process:aff=yes" \
torchrun --nproc_per_node 1 \
example_text_completion.py \
--ckpt_dir llama-2-7b/ \
--tokenizer_path tokenizer.model \
--max_seq_len 128 --max_batch_size 4

# 对话生成
bsub -Is -J llama -q normal_test -gpu "num=1:mode=exclusive_process:aff=yes" \
torchrun --nproc_per_node 1 \
example_chat_completion_test.py \
--ckpt_dir llama-2-7b-chat/ \
--tokenizer_path tokenizer.model \
--max_seq_len 512 --max_batch_size 4
```

# llama.cpp量化部署

## 1 克隆和编译llama.cpp

```sh
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make

# make编译遇到报错使用make clean重试
# 如果想使用GPU 得使用cuBLAS 一起编译 make LLAMA_CUBLAS=1 （目前没成功）
```

## 2 原版LLaMA模型转换为HF格式

![1698995968589](assets/1698995968589.png)

**原来的格式是.pth，想要量化需要转换成HuggingFace的格式**

```bash
# 使用transformers提供的脚本convert_llama_weights_to_hf.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py

# 将tokenizer.model、params.json和consolidated.xx.pth放入一个文件夹中作为输入文件
sudo python convert_llama_weights_to_hf.py --input_dir /home/mizzle/llm/quantization/transformers/models/codellm --model_size 7B --output_dir /home/mizzle/llm/quantization/transformers/models/codellm/codellm_hf_7b
```

输入文件夹：

![1698997519628](assets/1698997519628.png)

输出文件夹：

![1698997578049](assets/1698997578049.png)



## 3 生成量化版本模型并运行

将完整模型权重转换为**GGML**的FP16格式，生成文件路径为`models/llama-2-7b-hf/ggml-model-f16.bin`。进一步对FP16模型进行4-bit量化，生成量化模型文件路径为models/llama-2-7b-hf`/ggml-model-q4_0.bin`。 

目录结构如下：

```bash
- 7B/
  - consolidated.00.pth
  - params.json
- tokenizer.model
# 实际上tokenizer.model在7B的文件夹里或者parent dir里都行
- 7B/
  - consolidated.00.pth
  - params.json
  - tokenizer.model
```

将上述`.pth`模型权重转换为**GGML**的FP16格式，生成文件路径为`xx-models/7B/ggml-model-f16.bin`

```bash
# 最后输出是一个转换后的结果文件，无法设置output_dir,所以默认即可。
python convert.py ./models/codellm-7b/
```

### 量化

![1698999522483](assets/1698999522483.png)

```bash
# 观察性价比发现q4_1效果比较好 https://github.com/ggerganov/llama.cpp#perplexity-measuring-model-quality
./quantize ./models/codellm-7b/ggml-model-f16.gguf ./models/codellm-7b/ggml-model-q4_1.bin q4_1
# ./quantize ./models/codellm-7b/ggml-model-f16.gguf ./models/codellm-7b/ggml-model-q4_0.bin 2
```

### 加载并启动模型

```bash
./main -m models/codellm-7b/ggml-model-q4_1.bin --color -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.3
#./main -m models/codellm-7b/ggml-model-q4_0.bin --color -f prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.3

-ins 启动类ChatGPT对话交流的运行模式
-f 指定prompt模板，alpaca模型请加载prompts/alpaca.txt
-c 控制上下文的长度，值越大越能参考更长的对话历史（默认：512）
-n 控制回复生成的最大长度（默认：128）
-b 控制batch size（默认：8），可适当增加
-t 控制线程数量（默认：4），可适当增加
--repeat_penalty 控制生成回复中对重复文本的惩罚力度
--temp 温度系数，值越低回复的随机性越小，反之越大
--top_p, top_k 控制解码采样的相关参数
```









