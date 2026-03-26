# 🎯 Quick Start - CI/CD Setup

## 📝 What You Just Got

✅ `ecosystem.config.js` - PM2 process manager configuration  
✅ `.github/workflows/deploy.yml` - Auto-deploy on git push  
✅ `.env.example` - Template for environment variables  
✅ `setup-vm.sh` - One-command VM setup script  
✅ `DEPLOYMENT.md` - Complete deployment documentation

---

## ⚡ 3-Step Setup

### 1️⃣ On Your Azure VM (One-Time)

```bash
# SSH into your VM
ssh azureuser@YOUR_VM_IP

# Clone your repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Run the setup script
bash setup-vm.sh

# Edit .env and add your API key
nano .env
# Add: GOOGLE_API_KEY=your_actual_key

# Restart PM2
pm2 restart trix-chatbot
```

### 2️⃣ On GitHub (One-Time)

Go to: **Your Repo → Settings → Secrets → Actions**

Add these 4 secrets:

```
VM_HOST          = 20.123.45.67         (your VM IP)
VM_USER          = azureuser             (your SSH user)
VM_SSH_KEY       = -----BEGIN OPENSSH... (your private key)
ENV_FILE         = GOOGLE_API_KEY=abc... (your .env content)
```

**To get your SSH key:**

```bash
cat ~/.ssh/id_rsa  # Copy everything
```

**To get ENV_FILE content:**

```bash
cat .env  # Copy everything
```

### 3️⃣ Push and Watch Magic ✨

```bash
git add .
git commit -m "Setup CI/CD"
git push origin main
```

Go to GitHub → **Actions** tab → Watch it deploy!

---

## 🎮 Daily Workflow

```bash
# Make changes to your code
nano server2.py

# Commit and push
git add .
git commit -m "Updated feature X"
git push origin main
```

**That's it!** GitHub will automatically:

- Pull code on VM
- Install dependencies
- Restart the app
- App is live in ~30 seconds! 🚀

---

## 📊 Useful Commands

### On Your VM

```bash
# View running processes
pm2 status

# View live logs
pm2 logs trix-chatbot

# Restart app
pm2 restart trix-chatbot

# Stop app
pm2 stop trix-chatbot
```

### Check if app is running

```bash
curl http://localhost:3000/health
```

---

## ⚠️ Important Notes

### Security

- ✅ `.env` is in `.gitignore` (don't commit it!)
- ✅ Use GitHub Secrets for sensitive data
- ✅ Never share your SSH private key

### First-Time Issues

**"Port 5000 already in use"**

```bash
# The app now runs on port 3000
# Check your .env has: PORT=3000
```

**"PM2 not found"**

```bash
sudo npm install -g pm2
```

**"Permission denied"**

```bash
chmod 600 ~/.ssh/id_rsa  # On your local machine
```

---

## 🔗 Files Created

| File                           | Purpose                   |
| ------------------------------ | ------------------------- |
| `ecosystem.config.js`          | PM2 app configuration     |
| `.github/workflows/deploy.yml` | Auto-deployment workflow  |
| `.env.example`                 | Environment template      |
| `setup-vm.sh`                  | VM setup script           |
| `DEPLOYMENT.md`                | Full documentation        |
| `.gitignore`                   | Files to ignore (updated) |

---

## 📚 Read More

- Full docs: See [DEPLOYMENT.md](DEPLOYMENT.md)
- PM2 docs: https://pm2.keymetrics.io/
- GitHub Actions: https://docs.github.com/actions

---

## 🆘 Need Help?

1. Check `pm2 logs trix-chatbot`
2. Check GitHub Actions tab for deployment logs
3. Verify all secrets are set correctly
4. SSH into VM and check manually

---

**You're all set! 🎉 Push to `main` and watch it deploy!** 🚀
