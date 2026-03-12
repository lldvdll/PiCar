# MLiS2 VM Setup Guide — PiCar Kaggle Challenge

## 1. SSH into the Machine

```bash
ssh [uni username]@mlis2.nottingham.ac.uk
```

## 2. Install Anaconda

```bash
bash /shared/Anaconda3-2024.10-1-Linux-x86_64.sh
```
% Why not miniconda?

- Press ENTER to confirm default install location (`/home/[uni username]/anaconda3`)
- Type `yes` when asked to initialize conda
- Then run:

```bash
source ~/.bashrc
```

## 3. Create Conda Environment

```bash
conda create --name picar2 python=3.9 -y
conda activate picar2
```

## 4. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn opencv-python pillow tensorflow kaggle
```

## 5. Set Up Kaggle API

Get your API token from kaggle.com → Settings → API → Generate New Token, then:

```bash
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxx
```

To make it persist across sessions:

```bash
echo 'export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxx' >> ~/.bashrc
```

## 6. Download Competition Data

```bash
mkdir -p ~/picar-kaggle/data
kaggle competitions download -c machine-learning-in-science-ii-2026
unzip machine-learning-in-science-ii-2026.zip -d ~/picar-kaggle/data/
```
% fix this to save data into project, where scripts expect it

## 7. Set Up SSH Key for GitHub

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press Enter for all prompts (default location, no passphrase). Then copy the public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the entire output line, go to **github.com → Settings → SSH and GPG keys → New SSH key**, paste it in, and save.

Test the connection:

```bash
ssh -T git@github.com
```

Type `yes` when prompted about host authenticity. You should see: `Hi <username>! You've successfully authenticated...`

## 8. Clone the Repo & Copy Data

```bash
cd ~
git clone git@github.com:lldvdll/PiCar.git
cd PiCar
cp -r ~/picar-kaggle/data/ ./data/
```

**Watch for nested folders** — `cp -r` can create `data/data/`. Fix with:

```bash
mv data/data/* data/
rmdir data/data
```

Also check for double-nested image folders (`training_data/training_data/`).

## 9. Configure Git

```bash
git config --global user.email "your_email@example.com"
git config --global user.name "Ningqian"
```

Set remote to SSH (if cloned via HTTPS):

```bash
git remote set-url origin git@github.com:lldvdll/PiCar.git
```

---

## Reconnecting After Disconnect

Every time you reconnect to the VM:

```bash
source ~/.bashrc
conda activate picar2
cd ~/PiCar
```
