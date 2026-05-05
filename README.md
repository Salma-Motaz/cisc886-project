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

## 1.Project Overview

UbuntuAssist is a domain-specific conversational assistant fine-tuned on the Ubuntu Dialogue Corpus for the purpose of resolving technical support queries pertaining to the Ubuntu operating system. The system integrates a cloud-native, end-to-end machine learning pipeline deployed on Amazon Web Services (AWS), encompassing distributed data preprocessing via Apache Spark on EMR, supervised fine-tuning of a Llama 3B base model on Lightning AI, quantised model serving via Ollama on a private EC2 instance, and a browser-accessible chat interface delivered through OpenWebUI. All infrastructure is provisioned as code using Terraform, ensuring full reproducibility and version control.

\---

## 2.Repository Structure

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
├── dataset/
│   └── convert_to_parquet.py
│ 
├── output/
│   │    └── figures/
│   │
│   └── processed/
│        ├── train  
│        ├── teat    
│        └── val
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

### 7.1 Notebook 1 – Data Preparation (`notebook1_data_prep.ipynb`)

**Run this notebook first.** It performs the following operations:

1. Reads the raw Parquet files from the preprocessed S3 output.
2. Groups individual message rows into complete dialogue sequences.
3. Scores each dialogue for quality using heuristic criteria.
4. Filters the dataset to retain only high-quality samples.
5. Serialises three CSV files to disk: `train_ready.csv`, `valid_ready.csv`, and `test_ready.csv`.

### 7.2 Notebook 2 – Training and Evaluation (`notebook2_training_testing.ipynb`)

**Run this notebook second, using a fresh kernel** (Kernel → Restart before executing any cells).

**Prerequisites:**

* `train_ready.csv`, `valid_ready.csv`, and `test_ready.csv` must exist (produced by Notebook 1).
* `HF_TOKEN` must be configured in Lightning AI Secrets (HuggingFace → Settings → Access Tokens → New token with Write permission).
* The Meta Llama 3.2 licence must be accepted on HuggingFace.

**Pipeline steps:**

1. Load prepared CSV files and convert to HuggingFace `Dataset` format for use with `SFTTrainer`.
2. Load **Llama 3.2 3B Instruct** in 4-bit quantisation (QLoRA) via Unsloth.
3. Attach LoRA adapters (rank = 16, alpha = 32) to attention and MLP projection layers.
4. Evaluate the base (pre-training) model to establish a performance baseline using ROUGE-L, Relevance Rate, and a Classification Report.
5. Fine-tune the model for **3 epochs** using QLoRA with the following hyperparameters:

   * Learning rate: `1e-4` with cosine decay schedule
   * Batch size: 2, gradient accumulation steps: 4 (effective batch size = 8)
   * Optimiser: AdamW 8-bit
   * Sequence packing enabled for training throughput
6. Evaluate the fine-tuned model against the held-out test set using ROUGE-L, BERTScore, Accuracy, Precision, Recall, F1, and a full Classification Report.
7. Display qualitative side-by-side comparisons of base model versus fine-tuned model responses.
8. Save the LoRA adapter weights locally.
9. Export the model to GGUF format (`q4_k_m` quantisation) for deployment via Ollama.
10. Push the LoRA adapters and GGUF model to HuggingFace Hub.

> *Note: Training on 30,000 samples requires approximately 6 hours on a T4 GPU.

**Checkpoint recovery:** If the kernel restarts during training, re-run all cells except Cell 6, replacing `trainer.train()` with:

```python
trainer.train(resume_from_checkpoint=get_latest_checkpoint())
```

**Output files produced:**

|File / Directory|Description|
|-|-|
|`checkpoints/checkpoint-11250/`|Intermediate LoRA adapter checkpoint|
|`llama3b_ubuntu_lora/`|Final saved LoRA adapter weights|
|`llama3b_ubuntu_gguf.gguf`|Quantised GGUF model for Ollama|
|`evaluation_comparison.png`|Base vs. fine-tuned performance comparison plots|

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

