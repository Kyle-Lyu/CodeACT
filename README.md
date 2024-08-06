# 🐱 CodeACT: Code Adaptive Compute-efficient Tuning Framework for Code LLMs

[![Paper](https://img.shields.io/badge/Paper-arXiv?logo=arxiv&logoColor=%23B31B1B&label=arXiv&labelColor=%23f5f5dc&color=%23B31B1B)](https://arxiv.org/abs/2408.02193)

> We are planning to unveil the codes soon.

## Overview

Motivated by the need for more effective and efficient training, we propose the **Code** **A**daptive **C**ompute-efficient **T**uning (**CodeACT**) framework. CodeACT introduces the **C**omplexity and **D**iversity **A**ware **S**ampling (**CDAS**) method to select high-quality training data based on complexity and diversity, and the **Dynamic Pack** padding strategy to reduce computational resource usage by minimizing padding tokens during training. 

Experimental results demonstrate that ***CodeACT-DeepSeek-Coder-6.7B***, fine-tuned on only 40% of the EVOL-Instruct data, achieves an 8.6\% performance increase on HumanEval, reduces training time by 78%, and decreases peak GPU memory usage by 27%. 

## Components

### Complexity and Diversity Aware Sampling

<div align='center'>
<img alt="image" src='assets/CDAS.png'>
</div>

An overviw of our proposed CDAS method, including three steps from top to bottom. Step 1: Clustering the EVOL-Instruct dataset to form multiple clusters. Step 2: Computing the Instruction-Following Difficulty score by comparing the model's perplexity with and without instructions. Step 3: Sampling the top m\% instances from each re-ranked cluster to form a high-complexity sub-dataset that preserves data diversity. Finally, we use the selected data for fine-tuning to obtain CodeACT-Coder.

### Dynamic Pack

<div align='center'>
<img alt="image" src='assets/DynamicPack.png'>
</div>

Illustration of different padding strategies, where the blank squares represent padding tokens. Top: Traditional padding strategy aligns samples to the model's maximum input length, resulting in high computational resource consumption. Middle: Dynamic padding strategy reduces the number of padding tokens by aligning samples to the length of the longest sample in each batch. Bottom: Our proposed Dynamic Pack strategy sorts samples by length and concatenates multiple samples within a batch, further optimizing the utilization of the model's maximum input length and reducing padding tokens.

## Results

### RQ1: How does the CodeACT framework perform across different datasets and models?

The *CodeACT* column indicates whether the model was trained using our framework. The **bold** scores represent the best performance achieved using the same base model. The results highlight the efficiency gains achieved by CodeACT in terms of reduced training time and peak GPU memory usage, while maintaining or improving performance across various benchmarks.

<table>
  <thead>
    <tr>
      <th style="text-align: left;" rowspan="2"><strong>Model</strong></th>
      <th style="text-align: center;" rowspan="2"><strong>Size</strong></th>
      <th style="text-align: center;" rowspan="2"><strong>CodeACT</strong></th>
      <th style="text-align: center;" colspan="2"><strong>Efficiency</strong></th>
      <th style="text-align: center;" colspan="4"><strong>Benchmark (Pass@1 %)</strong></th>
    </tr>
    <tr>
      <th style="text-align: center;"><strong>Training Time</strong></th>
      <th style="text-align: center;"><strong>Peak GPU Memory</strong></th>
      <th style="text-align: center;"><strong>HumanEval</strong></th>
      <th style="text-align: center;"><strong>HumanEval+</strong></th>
      <th style="text-align: center;"><strong>MBPP</strong></th>
      <th style="text-align: center;"><strong>MBPP+</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;" colspan="8"><em>Models trained on OSS-Instruct dataset</em></td>
    </tr>
    <tr>
      <td style="text-align: left;" rowspan="2">CodeLlama</td>
      <td style="text-align: center;" rowspan="2">7B</td>
      <td style="text-align: center;">❌</td>
      <td style="text-align: center;">220 min</td>
      <td style="text-align: center;">64.28 GB</td>
      <td style="text-align: center;">50.6</td>
      <td style="text-align: center;">47.0</td>
      <td style="text-align: center;"><strong>63.2</strong></td>
      <td style="text-align: center;"><strong>51.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: center;">✔️</td>
      <td style="text-align: center;"><strong>63 min</strong></td>
      <td style="text-align: center;"><strong>33.21 GB</strong></td>
      <td style="text-align: center;"><strong>54.3</strong></td>
      <td style="text-align: center;"><strong>50.0</strong></td>
      <td style="text-align: center;">60.4</td>
      <td style="text-align: center;">50.4</td>
    </tr>
    <tr>
      <td style="text-align: left;" rowspan="2">CodeLlama</td>
      <td style="text-align: center;" rowspan="2">13B</td>
      <td style="text-align: center;">❌</td>
      <td style="text-align: center;">367 min</td>
      <td style="text-align: center;">69.17 GB</td>
      <td style="text-align: center;">58.5</td>
      <td style="text-align: center;"><strong>52.4</strong></td>
      <td style="text-align: center;"><strong>63.2</strong></td>
      <td style="text-align: center;"><strong>51.9</strong></td>
    </tr>
    <tr>
      <td style="text-align: center;">✔️</td>
      <td style="text-align: center;"><strong>109 min</strong></td>
      <td style="text-align: center;"><strong>59.62 GB</strong></td>
      <td style="text-align: center;"><strong>59.8</strong></td>
      <td style="text-align: center;"><strong>52.4</strong></td>
      <td style="text-align: center;"><strong>63.2</strong></td>
      <td style="text-align: center;">50.6</td>
    </tr>
    <tr>
      <td style="text-align: left;" rowspan="2">DeepSeek-Coder</td>
      <td style="text-align: center;" rowspan="2">6.7B</td>
      <td style="text-align: center;">❌</td>
      <td style="text-align: center;">184 min</td>
      <td style="text-align: center;">64.48 GB</td>
      <td style="text-align: center;">65.2</td>
      <td style="text-align: center;">61.0</td>
      <td style="text-align: center;"><strong>75.9</strong></td>
      <td style="text-align: center;"><strong>63.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: center;">✔️</td>
      <td style="text-align: center;"><strong>54 min</strong></td>
      <td style="text-align: center;"><strong>35.51 GB</strong></td>
      <td style="text-align: center;"><strong>68.3</strong></td>
      <td style="text-align: center;"><strong>61.6</strong></td>
      <td style="text-align: center;"><strong>75.9</strong></td>
      <td style="text-align: center;">61.7</td>
    </tr>
    <tr>
      <td style="text-align: center;" colspan="8"><em>Models trained on EVOL-Instruct dataset</em></td>
    </tr>
    <tr>
      <td style="text-align: left;" rowspan="2">CodeLlama</td>
      <td style="text-align: center;" rowspan="2">7B</td>
      <td style="text-align: center;">❌</td>
      <td style="text-align: center;">297 min</td>
      <td style="text-align: center;">50.65 GB</td>
      <td style="text-align: center;"><strong>54.3</strong></td>
      <td style="text-align: center;"><strong>50.0</strong></td>
      <td style="text-align: center;"><strong>60.7</strong></td>
      <td style="text-align: center;">48.6</td>
    </tr>
    <tr>
      <td style="text-align: center;">✔️</td>
      <td style="text-align: center;"><strong>68 min</strong></td>
      <td style="text-align: center;"><strong>38.25 GB</strong></td>
      <td style="text-align: center;">53.0</td>
      <td style="text-align: center;">47.0</td>
      <td style="text-align: center;"><strong>60.7</strong></td>
      <td style="text-align: center;"><strong>49.9</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;" rowspan="2">CodeLlama</td>
      <td style="text-align: center;" rowspan="2">13B</td>
      <td style="text-align: center;">❌</td>
      <td style="text-align: center;">468 min</td>
      <td style="text-align: center;">69.17 GB</td>
      <td style="text-align: center;">62.2</td>
      <td style="text-align: center;"><strong>56.7</strong></td>
      <td style="text-align: center;"><strong>63.2</strong></td>
      <td style="text-align: center;"><strong>52.9</strong></td>
    </tr>
    <tr>
      <td style="text-align: center;">✔️</td>
      <td style="text-align: center;"><strong>116 min</strong></td>
      <td style="text-align: center;"><strong>58.82 GB</strong></td>
      <td style="text-align: center;"><strong>64.0</strong></td>
      <td style="text-align: center;">55.5</td>
      <td style="text-align: center;">62.4</td>
      <td style="text-align: center;">51.6</td>
    </tr>
    <tr>
      <td style="text-align: left;" rowspan="2">DeepSeek-Coder</td>
      <td style="text-align: center;" rowspan="2">6.7B</td>
      <td style="text-align: center;">❌</td>
      <td style="text-align: center;">259 min</td>
      <td style="text-align: center;">50.75 GB</td>
      <td style="text-align: center;">58.5</td>
      <td style="text-align: center;">53.7</td>
      <td style="text-align: center;"><strong>71.4</strong></td>
      <td style="text-align: center;"><strong>58.1</strong></td>
    </tr>
    <tr>
      <td style="text-align: center;">✔️</td>
      <td style="text-align: center;"><strong>58 min</strong></td>
      <td style="text-align: center;"><strong>37.23 GB</strong></td>
      <td style="text-align: center;"><strong>67.1</strong></td>
      <td style="text-align: center;"><strong>59.8</strong></td>
      <td style="text-align: center;">69.9</td>
      <td style="text-align: center;"><strong>58.1</strong></td>
    </tr>
  </tbody>
</table>

### RQ2: How does the performance of models trained with CodeACT compare to other models?

The **bold** scores indicate the highest performance among models of the same size. The results show that models trained with CodeACT outperform their base models and achieve competitive results compared to other state-of-the-art open-source models. Additionally, our models demonstrate impressive efficiency in data utilization. Despite using fewer data samples, models trained with CodeACT achieve better performance compared to models that utilize a larger dataset.

<table>
  <thead>
    <tr>
      <th style="text-align: left;" rowspan="2"><strong>Model</strong></th>
      <th style="text-align: center;" rowspan="2"><strong>Size</strong></th>
      <th style="text-align: center;" rowspan="2"><strong>Base Model</strong></th>
      <th style="text-align: center;" rowspan="2"><strong>Data Type</strong></th>
      <th style="text-align: center;" rowspan="2"><strong>Data Num</strong></th>
      <th style="text-align: center;" colspan="4"><strong>Benchmark (Pass@1 %)</strong></th>
    </tr>
    <tr>
      <th style="text-align: center;"><strong>HumanEval</strong></th>
      <th style="text-align: center;"><strong>HumanEval+</strong></th>
      <th style="text-align: center;"><strong>MBPP</strong></th>
      <th style="text-align: center;"><strong>MBPP+</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;" colspan="9"><em>Closed-source Models</em></td>
    </tr>
    <tr>
      <td style="text-align: left;">Gemini-Pro-1.0</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">Proprietary</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">63.4</td>
      <td style="text-align: center;">55.5</td>
      <td style="text-align: center;">75.4</td>
      <td style="text-align: center;">61.4</td>
    </tr>
    <tr>
      <td style="text-align: left;">Claude-3-Opus</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">Proprietary</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">82.9</td>
      <td style="text-align: center;">77.4</td>
      <td style="text-align: center;"><strong>89.4</strong></td>
      <td style="text-align: center;"><strong>73.3</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">GPT-4-Turbo</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">Proprietary</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>85.4</strong></td>
      <td style="text-align: center;"><strong>81.7</strong></td>
      <td style="text-align: center;">85.7</td>
      <td style="text-align: center;"><strong>73.3</strong></td>
    </tr>
    <tr>
      <td style="text-align: center;" colspan="9"><em>Open-source Models</em></td>
    </tr>
    <tr style="background-color: #e6f2ff;">
      <td style="text-align: left;">CodeLlama</td>
      <td style="text-align: center;">34B</td>
      <td style="text-align: center;">Llama2</td>
      <td style="text-align: center;">Proprietary</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">51.8</td>
      <td style="text-align: center;">43.9</td>
      <td style="text-align: center;">65.4</td>
      <td style="text-align: center;">52.6</td>
    </tr>
    <tr style="background-color: #e6f2ff;">
      <td style="text-align: left;">WizardCoder-CL</td>
      <td style="text-align: center;">34B</td>
      <td style="text-align: center;">CodeLlama</td>
      <td style="text-align: center;">EVOL-Instruct</td>
      <td style="text-align: center;">78K</td>
      <td style="text-align: center;"><strong>73.2</strong></td>
      <td style="text-align: center;"><strong>64.6</strong></td>
      <td style="text-align: center;"><strong>73.2</strong></td>
      <td style="text-align: center;"><strong>59.9</strong></td>
    </tr>
    <tr style="background-color: #f0e6ff;">
      <td style="text-align: left;">StarCoder</td>
      <td style="text-align: center;">15B</td>
      <td style="text-align: center;">StarCoderBase</td>
      <td style="text-align: center;">Proprietary</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">34.1</td>
      <td style="text-align: center;">29.3</td>
      <td style="text-align: center;">55.1</td>
      <td style="text-align: center;">46.1</td>
    </tr>
    <tr style="background-color: #f0e6ff;">
      <td style="text-align: left;">CodeLlama</td>
      <td style="text-align: center;">13B</td>
      <td style="text-align: center;">Llama 2</td>
      <td style="text-align: center;">Proprietary</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">43.3</td>
      <td style="text-align: center;">36.6</td>
      <td style="text-align: center;">57.6</td>
      <td style="text-align: center;">46.9</td>
    </tr>
    <tr style="background-color: #f0e6ff;">
      <td style="text-align: left;">WizardCoder-SC</td>
      <td style="text-align: center;">15B</td>
      <td style="text-align: center;">StarCoder</td>
      <td style="text-align: center;">EVOL-Instruct</td>
      <td style="text-align: center;">78K</td>
      <td style="text-align: center;">56.7</td>
      <td style="text-align: center;">50.6</td>
      <td style="text-align: center;">59.6</td>
      <td style="text-align: center;">48.1</td>
    </tr>
    <tr style="background-color: #f0e6ff;">
      <td style="text-align: left;">CodeACT-CL (ours)</td>
      <td style="text-align: center;">13B</td>
      <td style="text-align: center;">CodeLlama</td>
      <td style="text-align: center;">EVOL-Instruct</td>
      <td style="text-align: center;">31K</td>
      <td style="text-align: center;"><strong>64.0</strong></td>
      <td style="text-align: center;"><strong>55.5</strong></td>
      <td style="text-align: center;"><strong>62.4</strong></td>
      <td style="text-align: center;"><strong>51.6</strong></td>
    </tr>
    <tr style="background-color: #e6ffe6;">
      <td style="text-align: left;">CodeLlama</td>
      <td style="text-align: center;">7B</td>
      <td style="text-align: center;">Llama 2</td>
      <td style="text-align: center;">Proprietary</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">39.0</td>
      <td style="text-align: center;">34.1</td>
      <td style="text-align: center;">58.1</td>
      <td style="text-align: center;">46.1</td>
    </tr>
    <tr style="background-color: #e6ffe6;">
      <td style="text-align: left;">DeepSeek-Coder-Base</td>
      <td style="text-align: center;">6.7B</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">Proprietary</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">47.6</td>
      <td style="text-align: center;">40.2</td>
      <td style="text-align: center;">69.2</td>
      <td style="text-align: center;">54.6</td>
    </tr>
    <tr style="background-color: #e6ffe6;">
      <td style="text-align: left;">WizardCoder-CL</td>
      <td style="text-align: center;">7B</td>
      <td style="text-align: center;">CodeLlama</td>
      <td style="text-align: center;">EVOL-Instruct</td>
      <td style="text-align: center;">78K</td>
      <td style="text-align: center;">50.6</td>
      <td style="text-align: center;">45.1</td>
      <td style="text-align: center;">58.5</td>
      <td style="text-align: center;">49.5</td>
    </tr>
    <tr style="background-color: #e6ffe6;">
      <td style="text-align: left;">CodeACT-CL (ours)</td>
      <td style="text-align: center;">7B</td>
      <td style="text-align: center;">CodeLlama</td>
      <td style="text-align: center;">EVOL-Instruct</td>
      <td style="text-align: center;">31K</td>
      <td style="text-align: center;">53.0</td>
      <td style="text-align: center;">47.0</td>
      <td style="text-align: center;">60.7</td>
      <td style="text-align: center;">49.9</td>
    </tr>
    <tr style="background-color: #e6ffe6;">
      <td style="text-align: left;">Magicoder-DS</td>
      <td style="text-align: center;">6.7B</td>
      <td style="text-align: center;">DeepSeek-Coder</td>
      <td style="text-align: center;">OSS-Instruct</td>
      <td style="text-align: center;">75K</td>
      <td style="text-align: center;">66.5</td>
      <td style="text-align: center;">60.4</td>
      <td style="text-align: center;">75.4</td>
      <td style="text-align: center;"><strong>61.9</strong></td>
    </tr>
    <tr style="background-color: #e6ffe6;">
      <td style="text-align: left;">CodeACT-DS (ours)</td>
      <td style="text-align: center;">6.7B</td>
      <td style="text-align: center;">DeepSeek-Coder</td>
      <td style="text-align: center;">OSS-Instruct</td>
      <td style="text-align: center;">30K</td>
      <td style="text-align: center;"><strong>68.3</strong></td>
      <td style="text-align: center;"><strong>61.6</strong></td>
      <td style="text-align: center;"><strong>75.9</strong></td>
      <td style="text-align: center;">61.7</td>
    </tr>
  </tbody>
</table>


## Citation

If you find CodeACT helpful, please cite it as follows:

```bibtex
@misc{lv2024codeactcodeadaptivecomputeefficient,
      title={CodeACT: Code Adaptive Compute-efficient Tuning Framework for Code LLMs}, 
      author={Weijie Lv and Xuan Xia and Sheng-Jun Huang},
      year={2024},
      eprint={2408.02193},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.02193}, 
}
```
