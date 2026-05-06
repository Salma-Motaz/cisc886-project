# Ubuntu Assistant: Fine-Tuned Large Language Model for Technical Issue Resolution

\---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [System Architecture](#system-architecture)
5. [Phase 1 – Infrastructure Setup (Terraform + AWS CLI)](#5-phase-1--infrastructure-setup-terraform--aws-cli)
6. [Phase 2 – Data Acquisition and Preprocessing (EMR + PySpark)](#6-phase-2--data-acquisition-and-preprocessing-emr--pyspark)
7. [Phase 3 – Model Fine-Tuning (QLoRA + Unsloth)](#7-phase-3--model-fine-tuning-qlora--unsloth)
8. [Phase 4 – Model Deployment (EC2 + Ollama + Open WebUI)](#8-phase-4--model-deployment-ec2--ollama--open-webui)
9. [Phase 5 – Infrastructure Teardown](#9-phase-5--infrastructure-teardown)
10. [AWS Cost Summary](#10-aws-cost-summary)
11. [Model Deployment Steps](#11-Model-Deployment-Steps)

\---

## 1\. Project Overview

UbuntuAssist is a domain-specific conversational assistant fine-tuned on the Ubuntu Dialogue Corpus for the purpose of resolving technical support queries pertaining to the Ubuntu operating system. The system integrates a cloud-native, end-to-end machine learning pipeline deployed on Amazon Web Services (AWS), encompassing distributed data preprocessing via Apache Spark on EMR, supervised fine-tuning of a Llama 3B base model on Lightning AI, quantised model serving via Ollama on a private EC2 instance, and a browser-accessible chat interface delivered through OpenWebUI. All infrastructure is provisioned as code using Terraform, ensuring full reproducibility and version control.

\---

## 2\. Repository Structure

```
repo/
.
├── Terraform_files/
│   ├── foundation/
│   │    ├── main.tf    # VPC, subnets, IGW, NAT Gateway, route tables, S3 VPC Endpoint
│   │    └── iam.tf     # IAM roles and instance profiles for EMR and EC2
│   │
│   ├── emr/
│   │    └── emr.tf     # EMR cluster configuration for Spark preprocessing 
│   │
│   ├── deployment/
│   │    └── deployment.tf # Bastion host and private LLM server (EC2 + Ollama + OpenWebUI)
│   │
│   ├── utilities/
│   │    ├── 25nplx-keypair      
│   │    └── 25nplx-keypair.pub 
│   │
│   └──preprocessing/
│        ├── bootstrap.sh             # EMR bootstrap script for Python dependency installation
│        └── ubuntu_preprocessing.py  # PySpark preprocessing script for the Ubuntu Dialogue Corpus
│   
├── notebooks/
│   ├── notebook1_data_prep.ipynb          # Lightning AI fine-tuning notebook (splitting the data)
│   ├── notebook2_training_testing.ipynb   # Lightning AI fine-tuning notebook (Llama 3B, LoRA, GGUF export)
│   └── convert_to_parquet.py
│ 
├── output/
│       └── figures/
│           ├── fig1_token_distribution.png
│           ├── fig2_turns_per_dialogue.png
│           ├── fig3_split_counts.png
│           └── fig4_time_gap.png
│
├── Project_Replication_Guideline/
│       └── Group_12_CISC886_Project_Replication_Guide.pdf  
│        
└── README.md

```

\---

## 3\. Prerequisites

### 3.1 Tools and Software

|Tool|Version Used|Installation|
|-|-|-|
|Terraform|v1.14.8|Via Chocolatey (Windows)|
|AWS CLI|v2.34.27|Official installer|
|Python|3.14.x|Required for preprocessing scripts|
|PowerShell|5.1+|Windows built-in|
|SSH client|Any|Windows OpenSSH or equivalent|

### 3.2 AWS Account Requirements

* An AWS account with programmatic access enabled.
* An IAM user with the **AdministratorAccess** policy attached.
* An active access key pair (Access Key ID + Secret Access Key).
* Default region: **us-east-1** (US East – N. Virginia).

### 3.3 External Accounts and Licences

* A **HuggingFace** account with a write-permission access token (stored in Lightning AI Secrets as `HF_TOKEN`).
* Acceptance of the **Meta Llama 3.2 licence** at: [huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
* Access to the **Ubuntu Dialogue Corpus** dataset: [https://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/](https://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/).

\---

## 4\. System Architecture

The deployed system is organised into six principal components forming a secure, sequentially ordered pipeline.

### 4.1 Architecture Diagram

```
Internet
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  VPC — 10.0.0.0/16                                          │
│                                                             │
│  ┌─ Public Subnet (10.0.1.0/24, us-east-1a) ───────────┐    │
│  │  Internet Gateway (25nplx-igw)                      │    │
│  │  NAT Gateway (Elastic IP, outbound only)            │    │
│  │  Bastion Host (t3.micro, Ubuntu 22.04, port 22)     │    │
│  └─────────────────────────────────────────────────────┘    │
│           │ SSH tunnel                                      │
│  ┌─ Private Subnet (10.0.2.0/24, us-east-1a) ──────────┐    │
│  │  LLM Server (t3.2xlarge, no public IP)              │    │
│  │    Ollama  → port 11434                             │    │
│  │    OpenWebUI (Docker) → port 8080                   │    │
│  │  EMR Cluster (m5.xlarge × 3, auto-terminates)       │    │
│  └─────────────────────────────────────────────────────┘    │
│  S3 VPC Endpoint (free, no NAT charges)                     │
│  IAM Roles (EMR service, EMR EC2, EC2 deployment)           │
└─────────────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
    S3 Bucket                 HuggingFace
  (25nplx-cisc886)       (GGUF model host)
         ▲
         │
   Lightning AI
  (fine-tuning)
```

### 4.2 Component Summary

|Component|Technology|Location|
|-|-|-|
|Networking|VPC, IGW, NAT Gateway, S3 Endpoint|AWS|
|Data Preprocessing|EMR 6.15.0, Spark 3.4.1, Hadoop 3.3.6|Private subnet|
|Fine-Tuning|Lightning AI, LoRA, Llama 3B|External (Lightning AI)|
|Model Serving|Ollama, GGUF Q4\_K\_M|Private EC2 (t3.2xlarge)|
|Web Interface|OpenWebUI (Docker)|Private EC2 (t3.2xlarge)|
|Access \& Security|Bastion host, SSH tunnelling, security groups|Public subnet|

\---

## 5\. Phase 1 – Infrastructure Setup (Terraform + AWS CLI)

### 5.1 Install Terraform via Chocolatey

Execute the following commands in an elevated PowerShell session:

```powershell
- Set-ExecutionPolicy Bypass -Scope Process -Force
- [System.Net.ServicePointManager]::SecurityProtocol = `
  [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
- iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
choco install terraform -y
terraform -version
```

Expected output: `Terraform v1.14.8 on windows_amd64`

### 5.2 Install and Configure the AWS CLI

1. Download and install the official AWS CLI package from [https://aws.amazon.com/cli/](https://aws.amazon.com/cli/).
2. Restart the terminal to refresh environment variables.
3. Verify the installation:

```bash
aws --version
# Expected: aws-cli/2.34.27 Python/3.14.3 Windows/11 exe/AMD64
```

4. Configure programmatic access using the IAM user credentials:

```bash
aws configure
# AWS Access Key ID:     <your-access-key-id>
# AWS Secret Access Key: <your-secret-access-key>
# Default region name:   us-east-1
# Default output format: json
```

5. Validate the configuration:

```bash
aws sts get-caller-identity
```

### 5.3 Generate the SSH Key Pair

Navigate to the `Terraform_files/utilities/` directory and generate a 4096-bit RSA key pair. When prompted for a passphrase, press **Enter** twice to proceed without one.

```powershell
ssh-keygen -t rsa -b 4096 -f "Terraform_files\utilities\25nplx-keypair"
```

Import the public key into AWS:

```bash
aws ec2 import-key-pair `
  --key-name "25nplx-keypair" `
  --public-key-material fileb://25nplx-keypair.pub
```

### 5.4 Provision the Foundation Infrastructure

Navigate to the `foundation/` directory and execute the Terraform workflow:

```bash
cd Terraform_files/foundation
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
```

**Outputs produced:**

|Output Variable|Example Value|
|-|-|
|`availability_zone`|us-east-1a|
|`ec2_deployment_profile_name`|25nplx-ec2-deployment-profile|
|`private_subnet_id`|subnet-099bb09528fbee1b8|
|`public_subnet_id`|subnet-06ab8b608fd0becd0|
|`vpc_id`|vpc-0962a0563e0d101bb|

\---

## 6\. Phase 2 – Data Acquisition and Preprocessing (EMR + PySpark)

### 6.1 Create the S3 Bucket

```bash
aws s3 mb s3://25nplx-cisc886-bucket --region us-east-1
aws s3 ls  # Verify bucket creation
```

### 6.2 Download and Convert the Dataset

1. Download the Ubuntu Dialogue Corpus (TSV format) from [https://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/](https://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/).
2. Convert the dataset from TSV to Parquet format using the provided conversion script:

```bash
cd downloads
python convert_to_parquet.py
```

The script processes approximately 16.5 million rows (\~140 minutes to complete) and saves the output as `ubuntu_raw.parquet` (\~735 MB).

### 6.3 Upload Assets to S3

Upload the converted dataset:

```bash
aws s3 cp ubuntu_raw.parquet s3://25nplx-cisc886-bucket/ubuntu_raw.parquet
```

Upload the PySpark preprocessing script and EMR bootstrap file:

```bash
aws s3 cp bootstrap.sh s3://25nplx-cisc886-bucket/scripts/bootstrap.sh
aws s3 cp ubuntu_preprocessing.py s3://25nplx-cisc886-bucket/scripts/ubuntu_preprocessing.py
```

Copy the generated key pair into the `emr/` directory:

```bash
cp Terraform_files/utilities/25nplx-keypair     Terraform_files/emr/
cp Terraform_files/utilities/25nplx-keypair.pub Terraform_files/emr/
```

### 6.4 Provision and Run the EMR Cluster

Navigate to the `emr/` directory and execute the Terraform workflow:

```bash
cd Terraform_files/emr
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
```

**Cluster configuration:** 1 Primary node + 2 Core nodes (m5.xlarge), EMR 6.15.0, Hadoop 3.3.6, Spark 3.4.1.

The cluster will automatically terminate upon job completion. Preprocessed output is written to `s3://25nplx-cisc886-bucket/output/processed/` (split into `train/`, `val/`, and `test/` partitions). Visualisation figures are saved to `s3://25nplx-cisc886-bucket/output/figures/`.

After the EMR job concludes, remove the EMR security group resources:

```bash
terraform destroy -auto-approve
```

\---

## 7\. Phase 3 – Model Fine-Tuning (QLoRA + Unsloth)

Fine-tuning is performed in two sequential Jupyter notebooks on **Lightning AI** (or any GPU-enabled environment with a T4 GPU or equivalent).

### 7.1 # Notebook 1 — Data Preparation & Quality Scoring

## Overview
This notebook processes the raw Ubuntu dialogue corpus (provided as Parquet files) into cleaned, scored, and filtered CSV files ready for fine‑tuning a Llama 3.2 model. It:

1. Groups individual dialogue turns into full conversations
2. Scores each dialogue for technical quality (Linux commands, resolution, length, etc.)
3. Filters out low‑quality or off‑topic dialogues
4. Saves three CSV files: `train_ready.csv`, `valid_ready.csv`, `test_ready.csv`

These outputs are used by **Notebook 2 (Fine‑Tuning + HuggingFace Push)**.

---

## Prerequisites
- Python 3.10 or higher
- Any machine with at least 8 GB RAM (16 GB recommended for full dataset)
- The raw dataset: a ZIP archive containing **three folders** (`train/`, `validation/`, `test/`) – each folder contains multiple Parquet part files (e.g., `part-00000-….snappy.parquet`)

---

## Step 1: Unzip and Prepare the Parquet Files

### 1.1 Unzip the dataset
```bash
unzip ubuntu_dialogue.zip   # adjust name to your actual zip file
```
This will create the folders: `train/`, `validation/`, `test/`.

### 1.2 Concatenate part files into single Parquet files
The notebook expects **single** Parquet files named `train.parquet`, `validation.parquet`, and `test.parquet` in the same directory as the notebook.

#### Using pandas (simple, works for moderate‑sized datasets)
```python
import pandas as pd
import glob

# Train
train_parts = glob.glob("train/*.parquet")
train_df = pd.concat([pd.read_parquet(f) for f in train_parts], ignore_index=True)
train_df.to_parquet("train.parquet")

# Validation
valid_parts = glob.glob("val/*.parquet")
valid_df = pd.concat([pd.read_parquet(f) for f in valid_parts], ignore_index=True)
valid_df.to_parquet("validation.parquet")

# Test
test_parts = glob.glob("test/*.parquet")
test_df = pd.concat([pd.read_parquet(f) for f in test_parts], ignore_index=True)
test_df.to_parquet("test.parquet")
```
After this step, you will have three files: `train.parquet`, `validation.parquet`, `test.parquet`.

---

## Step 2: Set Up the Python Environment

### 2.1 Install Dependencies
Run the first cell of the notebook, or manually install:
```bash
pip install fastparquet pandas numpy matplotlib
```

---

## Step 3: Run the Notebook

Execute the cells in order. Each cell is clearly labelled.

| Cell | Description | Approximate Time |
|------|-------------|------------------|
| 1‑2 | Install and imports | < 1 min |
| 3 | Load raw Parquet files | 1‑2 min |
| 4 | Group rows into full dialogues | 5‑10 min |
| 5‑6 | Score and filter dialogues | 5‑10 min |
| 7 | Visualise score distributions | < 1 min |
| 8 | Check token length distribution | < 1 min |
| 9 | Take random samples and save CSVs | < 1 min |
| 10 | Verify saved files | < 1 min |

### Important Parameters You Can Adjust
- `QUALITY_THRESHOLD = 40` – minimum composite score (0‑100) to keep a dialogue. Increase for higher quality, decrease for more data.
- `MAX_TRAIN = 30000`, `MAX_VALID = 3000`, `MAX_TEST = 500` – number of dialogues to randomly sample from the filtered set.
- `MAX_TURNS = 5` – keep only the last `MAX_TURNS` of each dialogue (the resolution part is most valuable).

---

## Step 4: Output Files

After successful execution, you will find:

| File | Description | Typical Size |
|------|-------------|--------------|
| `train_ready.csv` | Training dialogues (Llama 3.2 chat format) | ~35 MB (30,000 dialogues) |
| `valid_ready.csv` | Validation dialogues | ~2.8 MB (3,000 dialogues) |
| `test_ready.csv` | Test dialogues | ~0.5 MB (500 dialogues) |
| `dialogue_quality_scores.png` | Diagnostic plots (score distribution, length, etc.) | – |

These CSV files are **directly consumable** by **Notebook 2** – no further preprocessing required.

---

## Next Steps

After this notebook finishes, proceed to **Notebook 2** to fine‑tune Llama 3.2 3B Instruct on the prepared CSV files.

### 7.2 Notebook 2 — Fine-Tuning + HuggingFace Push

## Overview
This notebook fine-tunes Llama 3.2 3B Instruct using QLoRA on an Ubuntu dialogue corpus, evaluates the model, and pushes the results to HuggingFace Hub.

## Prerequisites

### Hardware Requirements
- **GPU**: Tesla T4 (15.6 GB VRAM) or equivalent with at least 16GB VRAM
- **RAM**: 16GB minimum
- **Storage**: 20GB free space

### Software Requirements
- Python 3.10 or higher
- CUDA-capable GPU with drivers installed

## Step 1: Environment Setup

### 1.1 Create a Virtual Environment (Recommended)
```bash
python -m venv ubuntu-llm-env
source ubuntu-llm-env/bin/activate  # On Windows: ubuntu-llm-env\Scripts\activate
```

### 1.2 Install Dependencies
The notebook will automatically install Unsloth, but you can pre-install:
```bash
pip install -U "unsloth[colab-new]" torch transformers datasets trl rouge-score scikit-learn pandas matplotlib
```

### 1.3 Verify GPU Availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Step 2: Prepare Data Files

Ensure the following CSV files from **Notebook 1** are in the same directory as this notebook:
- `train_ready.csv` (training data)
- `valid_ready.csv` (validation data)
- `test_ready.csv` (test data)

Each CSV should contain a `full_text` column with formatted dialogue data.

## Step 3: Run the Notebook

### 3.1 Execute Cells in Order

| Cell | Action | Estimated Time |
|------|--------|----------------|
| 1-2 | Install dependencies & imports | 1-2 minutes |
| 3 | Load CSV → HF Dataset | < 1 minute |
| 4 | Load Llama 3.2 + QLoRA | 2-3 minutes |
| 5 | Baseline evaluation (100 samples) | ~11 minutes |
| 6 | Train model (1000 steps) | ~60 minutes |
| 7 | Plot training curve | < 1 minute |
| 8 | Fine-tuned evaluation (100 samples) | ~1.5 minutes |
| 9 | Save LoRA adapter locally | < 1 minute |
| 10 | Export to GGUF (optional) | ~16 minutes |
| 11 | Push to HuggingFace Hub (optional) | 5-10 minutes |

### 3.2 Key Configuration Parameters

The notebook uses these training parameters:
```python
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
r = 16 (LoRA rank)
lora_alpha = 32
max_steps = 1000
learning_rate = 2e-4
batch_size = 2
gradient_accumulation_steps = 4
```

## Step 4: Using the Trained Model

### 4.1 Generate Responses (Within Notebook)

After training, use the `generate_response()` function:

```python
# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='checkpoints/checkpoint-800',
    max_seq_length=2048,
    load_in_4bit=True,
)

# Generate response
response = generate_response("How do I check disk usage in Ubuntu?")
print(response)
```

### 4.2 Load Saved LoRA Adapter

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Llama-3.2-3B-Instruct',
    max_seq_length=2048,
    load_in_4bit=True,
)

# Load your LoRA adapter
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
)
```

### 4.3 Run with Ollama (GGUF Export)

If you exported to GGUF:

```bash
# On EC2 or local machine with Ollama
ollama create ubuntu-assistant -f Modelfile
ollama run ubuntu-assistant
```

Or pull directly from HuggingFace:
```bash
ollama pull hf.co/Rodina222/ubuntu-dialogue-llama3b-gguf
```

## Step 5: Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `MAX_SEQ_LENGTH` to 1024 or `batch_size` to 1 |
| Missing CSV files | Run Notebook 1 first to generate train_ready.csv, valid_ready.csv, test_ready.csv |
| CUDA out of memory | Close other applications, clear cache with `torch.cuda.empty_cache()` |
| Slow training | Ensure GPU is being used; check with `nvidia-smi` |
| HuggingFace upload fails | Verify HF_TOKEN is correct and has write permissions |

### Memory Optimization Tips
- Enable `use_gradient_checkpointing = 'unsloth'` (already configured)
- Use 4-bit quantization (already enabled)
- Reduce batch size to 1 if needed

## Step 6: Expected Outputs

After running the notebook, you should have:

| Output File | Description |
|-------------|-------------|
| `checkpoints/` | Training checkpoints (step 100, 200, ..., 1000) |
| `llama3b_ubuntu_lora/` | LoRA adapter files (~115 MB) |
| `training_curve.png` | Loss plot |
| `llama3b_ubuntu_gguf_gguf/` | GGUF quantized model (~2 GB, optional) |

## Step 7: Evaluation Metrics Reference

Base model (before fine-tuning):
- ROUGE-L F1: 0.061
- Relevance Rate: 99.0%
- Accuracy: 54.0%

Fine-tuned model (step 800):
- ROUGE-L F1: 0.197
- Relevance Rate: 51.0%
- Accuracy: 64.0%

## Notes

- **Start with a fresh kernel** (Kernel → Restart) before running to ensure clean RAM
- The notebook uses `nrows=10000` for training and `nrows=2000` for validation to save time
- For production use, increase these to full dataset sizes
- The LoRA adapter is only 115 MB, making it easy to share and deploy
- GGUF export is optional but recommended for Ollama deployment on EC2

\---

## 8\. Phase 4 – Model Deployment (EC2 + Ollama + Open WebUI)

### 8.1 Provision the Deployment Infrastructure

Navigate to the `deployment/` directory and execute the Terraform workflow:

```bash
cd Terraform_files/deployment
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
```

**Outputs produced:**

|Output Variable|Example Value|
|-|-|
|`bastion_public_ip`|34.205.87.101|
|`ec2_private_ip`|10.0.2.163|

### 8.2 Configure the SSH Private Key

Rename the private key file to the `.pem` extension and restrict its permissions to the current user only:

```powershell
# Store the key path
$keyPath = "....\Terraform_files\utilities\25nplx-keypair.pem"

# Remove inherited permissions and grant read-only access to the current user
icacls $keyPath /inheritance:r
icacls $keyPath /grant:r "$($env:USERNAME):(R)"
icacls $keyPath /remove "NT AUTHORITY\Authenticated Users"
icacls $keyPath /remove "Everyone"
```

### 8.3 Connect via the Bastion Host

**SSH into the bastion host:**

```bash
ssh -i "...\Terraform_files\utilities\25nplx-keypair.pem" ubuntu@<bastion_public_ip>
```

**Transfer the private key to the bastion host:**

```bash
ssscp -i "...\Terraform_files\utilities\25nplx-keypair.pem" "....\Terraform_files\utilities\25nplx-keypair.pem" ubuntu@ bastion_public_ip:~/.ssh/25nplx-keypair.pem
```

**Set correct permissions on the bastion host, then SSH into the private EC2 instance:**

```bash
chmod 400 ~/.ssh/25nplx-keypair.pem
ssh -i ~/.ssh/25nplx-keypair.pem ubuntu@ec2_private_ip
```

### 8.4 Verify the Deployment on the Private EC2 Instance

**Step 1 – Confirm bootstrap script completion:**

```bash
sudo tail -f /var/log/user-data.log
# Press Ctrl+C to exit once "Setup complete!" is visible
```

**Step 2 – Verify Ollama is operational:**

```bash
systemctl status ollama
curl http://localhost:11434/api/tags
# Press Ctrl+C to exit
```

**Step 3 – Test model inference:**

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "25nplx-ubuntu-assistant-llama3b-chatbot",
  "prompt": "How do I check disk usage in Ubuntu?",
  "stream": false
}'
```

**Step 4 – Grant Docker permissions:**

```bash
sudo usermod -aG docker ubuntu
newgrp docker
docker ps
```

### 8.5 Access the Open WebUI Interface

In a **new local terminal**, create an SSH tunnel to forward the Open WebUI port:

```bash
ssh -i "...\Terraform_files\utilities\ 25nplx-keypair.pem" -L 8080: ec2_private_ip:8080 ubuntu@bastion_public_ip -N
```

Open a browser and navigate to:

```
http://localhost:8080/
```

The Ubuntu Assistant chatbot will be accessible through the Open WebUI interface, with the model `25nplx-ubuntu-assistant-llama3b-chatbot:latest` available for selection.

\---

## 9\. Phase 5 – Infrastructure Teardown

Perform teardown in the following order to avoid dependency conflicts.

**Step 1 – Terminate the SSH tunnel and exit all remote sessions:**  
Press `Ctrl+C` in the tunnel terminal, then type `exit` in each SSH session until all connections are closed.

**Step 2 – Destroy the deployment infrastructure:**

```bash
cd Terraform_files/deployment
terraform destroy -auto-approve
```

**Step 3 – Destroy the EMR security groups (if not already destroyed):**

```bash
cd Terraform_files/emr
terraform destroy -auto-approve
```

**Step 4 – Destroy the foundation infrastructure:**

```bash
cd Terraform_files/foundation
terraform destroy -auto-approve
```

Confirm final output: `Destroy complete! Resources: 19 destroyed.`

\---

## 10\. AWS Cost Summary

The following table summarises the approximate AWS expenditure incurred during a single end-to-end execution of this project.

|Service|Instance / Resource Type|Approximate Usage|Estimated Cost (USD)|
|-|-|-|-|
|EC2 (LLM Server)|t3.2xlarge|\~0.5 hours|\~$0.08|
|EC2 (Bastion Host)|t3.micro|\~0.5 hours|\~$0.01|
|EMR|3× m5.xlarge|\~0.25 hours|\~$0.60|
|NAT Gateway|Data transfer (Docker + model pull)|\~3 GB|\~$0.14|
|S3 Storage|Standard storage|\~1.5 GB|\~$0.04|
|Elastic IP|Static IP allocation|\~0.5 hours|\~$0.01|
|**Total**|||**\~$0.88**|

> All costs are estimates based on us-east-1 on-demand pricing as of April 2026. Actual costs may vary depending on usage duration and data transfer volumes.

\---

## 11\. Model Deployment Steps

### 11.1 Clone the Repository

```bash
git clone https://github.com/<your-username>/ubuntu-assistant.git
cd ubuntu-assistant
```

### 11.2 Configure AWS Credentials (IAM User)

```bash
aws configure
aws sts get-caller-identity
```

### 11.3 Generate SSH Key Pair and Import to AWS

```bash
ssh-keygen -t rsa -b 4096 -f "your_teraform_\Terraform_files\25nplx-keypair".

aws ec2 import-key-pair 
  --key-name "25nplx-keypair" 
  --public-key-material fileb:Terraform_files/utilities/25nplx-keypair.pub
```

### 11.4 Deploy Foundation Infrastructure (Terraform)

```bash
cd Terraform_files/foundation
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
```

### 11.5 Deploy Model Serving Infrastructure

```bash
cd Terraform_files/deployment
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
```

**Outputs produced:**

|Output Variable|Example Value|
|-|-|
|`bastion_public_ip`|34.205.87.101|
|`ec2_private_ip`|10.0.2.163|

### 11.6 Configure the SSH Private Key

Rename the private key file to the `.pem` extension and restrict its permissions to the current user only:

```powershell
# Store the key path

$keyPath = "....\Terraform_files\utilities\25nplx-keypair.pem"
# Remove inherited permissions and grant read-only access to the current user
icacls $keyPath /inheritance:r
icacls $keyPath /grant:r "$($env:USERNAME):(R)"
icacls $keyPath /remove "NT AUTHORITY\Authenticated Users"
icacls $keyPath /remove "Everyone"

```

### 11.7 Connect via the Bastion Host

**SSH into the bastion host:**

```bash
ssh -i "...\Terraform_files\utilities\25nplx-keypair.pem" ubuntu@<bastion_public_ip>
```

**Transfer the private key to the bastion host:**

```bash
sscp -i "...\Terraform_files\utilities\25nplx-keypair.pem" "....\Terraform_files\utilities\25nplx-keypair.pem" ubuntu@ bastion_public_ip:~/.ssh/25nplx-keypair.pem
```

**Set correct permissions on the bastion host, then SSH into the private EC2 instance:**

```bash
chmod 400 ~/.ssh/25nplx-keypair.pem
ssh -i ~/.ssh/25nplx-keypair.pem ubuntu@ec2_private_ip
```

### 11.8 Verify the Deployment on the Private EC2 Instance

**Step 1 – Confirm bootstrap script completion:**

```bash
sudo tail -f /var/log/user-data.log
# Press Ctrl+C to exit once "Setup complete!" is visible
```

**Step 2 – Verify Ollama is operational:**

```bash
systemctl status ollama
curl http://localhost:11434/api/tags
# Press Ctrl+C to exit
```

**Step 3 – Test model inference:**

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "25nplx-ubuntu-assistant-llama3b-chatbot",
  "prompt": "How do I check disk usage in Ubuntu?",
  "stream": false
}'
```

**Step 4 – Grant Docker permissions:**

```bash
sudo usermod -aG docker ubuntu
newgrp docker
docker ps
```

### 11.9 Access the Open WebUI Interface

In a **new local terminal**, create an SSH tunnel to forward the Open WebUI port:

```bash
ssh -i "...\Terraform_files\utilities\ 25nplx-keypair.pem" -L 8080: ec2_private_ip:8080 ubuntu@bastion_public_ip -N
```

Open a browser and navigate to:

```
http://localhost:8080/
```

The Ubuntu Assistant chatbot will be accessible through the Open WebUI interface, with the model `25nplx-ubuntu-assistant-llama3b-chatbot:latest` available for selection.

\---

### 11.10 Infrastructure Teardown

**Step 1 – Destroy Deployment Layer**

```bash
cd Terraform_files/deployment
terraform destroy -auto-approve
```

**Step 2 – Destroy Foundation Infrastructure**

```bash
cd Terraform_files/foundation
terraform destroy -auto-approve
```

Final output:

```
Destroy complete! Resources: 19 destroyed.
```

