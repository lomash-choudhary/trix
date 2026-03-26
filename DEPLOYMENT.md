# 🚀 Deployment Guide - Trix Chatbot

This guide will help you set up automated CI/CD deployment for your Flask chatbot using GitHub Actions and PM2.

---

## 📋 Prerequisites

- Azure VM with Ubuntu/Linux
- SSH access to your VM
- Node.js and npm installed on VM
- Python 3 and venv installed on VM
- Git repository on GitHub

---

## 🛠️ One-Time Setup on Azure VM

### 1. Install PM2 (Process Manager)

```bash
sudo npm install -g pm2
```

### 2. Clone Your Repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 3. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Create Logs Directory

```bash
mkdir -p logs
```

### 5. Update ecosystem.config.js

Edit the `cwd` path in `ecosystem.config.js` to match your VM path:

```javascript
cwd: "/home/azureuser/YOUR_FOLDER_NAME",  // Update this!
```

### 6. Initial PM2 Start

```bash
pm2 start ecosystem.config.js
pm2 save
pm2 startup  # Follow the command it gives you
```

### 7. Check PM2 Status

```bash
pm2 status
pm2 logs trix-chatbot
```

---

## 🔐 GitHub Secrets Setup

Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these secrets:

### Required Secrets:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `VM_HOST` | Your Azure VM IP address | `20.123.45.67` |
| `VM_USER` | SSH username | `azureuser` |
| `VM_SSH_KEY` | Your **private** SSH key | `-----BEGIN OPENSSH PRIVATE KEY-----...` |
| `VM_PORT` | SSH port (optional, default 22) | `22` |
| `ENV_FILE` | Your complete .env file content | See below ⬇️ |

---

## 📝 Creating the ENV_FILE Secret

### Step 1: Copy your local .env file content

Your `.env` file should look like this:

```bash
GOOGLE_API_KEY=your_actual_api_key_here
DOCUMENT_PATH=knowledge.txt
HOST=0.0.0.0
PORT=3000
DEBUG=false
```

### Step 2: Add to GitHub Secrets

1. Go to GitHub repo → Settings → Secrets → Actions
2. Click **New repository secret**
3. Name: `ENV_FILE`
4. Value: Paste your entire .env file content (all lines)
5. Click **Add secret**

---

## 🔑 Getting Your SSH Private Key

On your local machine (where you SSH from):

```bash
cat ~/.ssh/id_rsa  # or id_ed25519
```

Copy the **entire output** including:
- `-----BEGIN OPENSSH PRIVATE KEY-----`
- All the middle content
- `-----END OPENSSH PRIVATE KEY-----`

Paste this as the `VM_SSH_KEY` secret.

---

## 🚀 How Deployment Works

### Automatic Deployment Flow:

1. You push code to `main` branch
2. GitHub Actions triggers
3. Connects to your VM via SSH
4. Pulls latest code
5. Writes `.env` file from GitHub secret
6. Activates venv
7. Installs dependencies
8. Restarts PM2 process
9. ✅ App is live!

### Trigger Deployment:

```bash
git add .
git commit -m "Deploy changes"
git push origin main
```

Watch the deployment:
- Go to GitHub → Actions tab
- Click on the running workflow

---

## 📊 Useful PM2 Commands

### On Your VM:

```bash
# View status
pm2 status

# View logs (live)
pm2 logs trix-chatbot

# View last 100 lines
pm2 logs trix-chatbot --lines 100

# Restart app
pm2 restart trix-chatbot

# Stop app
pm2 stop trix-chatbot

# Delete app from PM2
pm2 delete trix-chatbot

# Monitor (interactive dashboard)
pm2 monit
```

---

## 🔧 Manual Deployment (if needed)

SSH into your VM and run:

```bash
cd /home/azureuser/trikon-2  # Your project path
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
pm2 restart trix-chatbot
```

---

## 🐛 Troubleshooting

### Port Already in Use

If you get "Port 3000 already in use":

```bash
pm2 list  # Check what's running
pm2 stop trix-chatbot
pm2 start ecosystem.config.js
```

### Check if process is running

```bash
pm2 status
pm2 logs trix-chatbot --lines 50
```

### GitHub Action Fails

1. Check the Actions tab for error messages
2. Verify all secrets are set correctly
3. Check SSH connection:
   ```bash
   ssh -i ~/.ssh/id_rsa azureuser@YOUR_VM_IP
   ```

### .env Not Loading

Make sure your `ENV_FILE` secret contains the complete content:

```
GOOGLE_API_KEY=abc123
DOCUMENT_PATH=knowledge.txt
PORT=3000
```

No extra quotes or formatting!

---

## 🔒 Security Tips

✅ **DO:**
- Keep `.env` in `.gitignore`
- Use GitHub Secrets for sensitive data
- Restrict SSH key permissions: `chmod 600 ~/.ssh/id_rsa`
- Use strong API keys

❌ **DON'T:**
- Commit `.env` to GitHub
- Share your SSH private key
- Push API keys in code
- Use default passwords

---

## 📈 Next Level (Optional)

### Add SSL/HTTPS with Nginx

```bash
sudo apt install nginx certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

### Zero-Downtime Deployments

Update `ecosystem.config.js`:

```javascript
instances: 2,  // Run 2 instances
exec_mode: "cluster",
```

### Auto-Rollback on Failure

Add health checks to GitHub Actions workflow.

---

## 📞 Need Help?

Common issues:
1. **"port already in use"** → Check PM2 and kill old processes
2. **"connection refused"** → Check VM firewall and NSG rules
3. **"permission denied"** → Check SSH key and VM user permissions

Check logs:
```bash
pm2 logs trix-chatbot --lines 100
```

---

## ✅ Checklist

Before first deployment, ensure:

- [ ] PM2 installed on VM
- [ ] Repository cloned on VM
- [ ] Virtual environment created
- [ ] `ecosystem.config.js` path updated
- [ ] All GitHub Secrets added
- [ ] `.env` is in `.gitignore`
- [ ] SSH key has correct permissions
- [ ] VM ports 22 and 3000 open in NSG

---

**🎉 You're ready for automatic deployments!**

Push to `main` and watch the magic happen! 🚀
