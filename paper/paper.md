
### Abstract




We present a continuous diffusion language modeling framework for the BabyLM
2026 Challenge. Our approach formulates language modeling in a continuous
embedding space and trains the model with a flow-matching objective, allowing
the model to learn a denoising dynamics over latent token representations rather
than over discrete token states.

We evaluate our system on the BabyLM benchmark suite in the 100M-word track,
using only the data permitted by the challenge. Across the official evaluation
tasks, our model outperforms the strongest official baseline and also surpasses
the previous 100M-track Masked Diffusion Language Model winner. These results
provide evidence that continuous diffusion can be a strong alternative to
discrete masked diffusion for data-constrained language modeling, and suggest
that flow-matching objectives are a promising direction for sample-efficient
language model pretraining.






### 1. Introduction







### Reseach Question




### 2. Related Work




ELF：

有点

limitaion for elf paper: 这篇论文的消融实验直接证明了，连续语言扩散模型能够用极少的数据和步数取得成功的关键，在于**直接白嫖商业大模型（如 T5）已经在大规模语料上预训练得极其完美的、具备上下文高级特征的连续语义流形空间** 。因此使用scratch encoder的方法，得到的结论更有说服力。本文主要探讨，Scratch encoder和 training epochs 在不同规模数据下的配比关系。




### 3. Methodology

BabyLM text → train tokenizer → tokenize corpus → train scratch encoder → freeze encoder → train ELF



Pretraining

architecture



Dataset




evaluation





方法:

We train the tokenizer from scratch using only the permitted BabyLM training corpus. The language model is subsequently trained for 10 epochs on the same corpus.



### 4. Result














### 5. Conclusions


连续扩散模型非常具有潜力






##### 5.1 Limitation



没有scale up，仅仅停留在100M


架构有改进空间




### Acknowledgements





### References








1. babaylm evaluation天然对mask更友好，而对生成更不友好。然后语言在现实中应该更注重生成能力，而不是填空能力
