# EDMR

![Methodology](figures/methodology.png)

EDMR is a novel methodology designed to improve Large Language Model reasoning capability in multimodal QA settings, where integration of information from multiple sources is required to correctly answer questions. 

EDMR is the first approach that quantifies and traces the contribution of each modality to the reduction of uncertainty over possible answer choices.

We evaluate EDMR using both closed-source and open-source LLMs. Our method improves state-of-the-art performance, outperforming both specialised systems and existing prompting strategies.

---

# Results

![Results](figures/results.png)

<table>
  <tr>
    <td align="center">
      <img src="figures/global_entropy_boxplot.png" width="400"><br>
      <em>Entropy lowering when adding modality</em>
    </td>
    <td align="center">
      <img src="figures/boxplot_tp_fp.png" width="400"><br>
      <em>Entropy TP vs Entropy FP</em>
    </td>
  </tr>
</table>




## 🚀 Installation

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/EMezzi/LEGuidance.git
cd LEGuidance
pip install -r requirements.txt
pip install -e .
```
2. Set up API keys and store environment variables in .env file

| Service     | Environment Variable    | Description                  | Example     |
|-------------|-------------------------|------------------------------|-------------|
| OpenAI      | `OPENAI_API_KEY`        | API key for OpenAI models    | `sk-...`    |
| AWS Bedrock | `AWS_ACCESS_KEY_ID`     | AWS access key ID            | `AKIA...`   |
| AWS Bedrock | `AWS_SECRET_ACCESS_KEY` | AWS secret access key        | `********`  |
| AWS Bedrock | `AWS_DEFAULT_REGION`    | AWS region (e.g., us-east-1) | `us-east-1` |

3. Download the datasets MultimodalQA and ManymodalQA and set the paths

`DATA_ROOT`=/path/to/multimodalqa_files

`IMAGE_DIR`='$DATA_ROOT/images'
`TEXT_DIR`='$DATA_ROOT/texts'
`TABLE_DIR`='$DATA_ROOT/tables'
`FINAL_DATASET_IMAGES`='$DATA_ROOT/final_dataset_images'

`ASSOCIATION_VALIDATION`='$DATA_ROOT/association_validation'

`CRITERIA_VALIDATION`='./results/multimodalqa/le/validation/criteria_extraction'
`ANSWERS_VALIDATION`='./results/multimodalqa/le/validation/QA_Answers'

`ANSWERS_VALIDATION_DP`='./results/multimodalqa/dp/validation/QA_Answers'
`ANSWERS_VALIDATION_COT`='./results/multimodalqa/cot/validation/QA_Answers'
`ANSWERS_VALIDATION_PP`='./results/multimodalqa/pp/validation/QA_Answers'

# Use
▶️ Running Experiments
The main script supports multiple datasets, models, and reasoning approaches.
Basic usage

| Argument     | Description        | Options                        | Default        |
|--------------|--------------------|--------------------------------|----------------|
| `--dataset`  | Dataset to use     | `multimodalqa`, `manymodalqa`  | `multimodalqa` |
| `--approach` | Reasoning method   | `dp`, `cot`, `pp`, `le`, `all` | `all`          |
| `--models`   | Models to evaluate | list of model IDs              | `["gpt-5.2"]`  |
| `--backend`  | API backend        | `openai`, `bedrock`            | `openai`       |


```bash
python main.py --dataset multimodalqa --setting validation --approach all
```

## Examples
Run all approaches with GPT-5.2:
```bash
python main.py --models gpt-5.2
```
Run only Chain-of-Thought:
```bash
python main.py --approach cot
```
Run LE pipeline (criteria + entropy analysis):
```bash
python main.py --approach le
```
Run Bedrock models:
```bash
python main.py --backend bedrock
```
