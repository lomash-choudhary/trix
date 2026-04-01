# 🚀 Deploying TRIX Chatbot to Render

## Quick Start (Recommended Method)

### Step 1: Push Your Code to GitHub

```bash
git add .
git commit -m "Configure for Render deployment"
git push origin main
```

### Step 2: Create a Render Account

1. Go to [render.com](https://render.com)
2. Sign up with your GitHub account

### Step 3: Deploy Using Blueprint (render.yaml)

1. In Render dashboard, click **"New +"** → **"Blueprint"**
2. Connect your GitHub repository (trikon-2)
3. Render will automatically detect `render.yaml`
4. Click **"Apply"**

### Step 4: Set Environment Variables

After deployment starts, go to your service settings and add:

**Required Environment Variables:**

- `GROQ_API_KEY` → Your Groq API key (get from [console.groq.com](https://console.groq.com))
- `GOOGLE_API_KEY` → Your Google API key (for embeddings)

**Optional (already set in render.yaml):**

- `DOCUMENT_PATH` → `knowledge.txt`
- `EMBEDDING_MODEL` → `models/gemini-embedding-001`
- `GROQ_MODEL` → `llama-3.3-70b-versatile`

### Step 5: Verify Deployment

Your app will be live at: `https://trix-chatbot.onrender.com` (or similar)

Test with:

```bash
# Health check
curl https://your-app.onrender.com/health

# Initialize
curl -X POST https://your-app.onrender.com/initialize

# Ask a question
curl -X POST https://your-app.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Trikon?"}'
```

---

## Alternative: Manual Web Service Creation

If you prefer manual setup instead of Blueprint:

1. **New Web Service**
   - Dashboard → "New +" → "Web Service"
   - Connect your GitHub repo

2. **Configure Service**
   - **Name**: `trix-chatbot`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python server2.py`
   - **Plan**: Free (or paid for better performance)

3. **Environment Variables** (same as above)

4. **Deploy!**

---

## Using GitHub Actions (Optional)

The workflow is already updated in `.github/workflows/deploy.yml`.

To use it:

1. **Get your Render Deploy Hook:**
   - Go to your Render service → Settings
   - Scroll to "Deploy Hook"
   - Copy the URL (looks like: `https://api.render.com/deploy/srv-xxxxx?key=yyyyy`)

2. **Add to GitHub Secrets:**
   - Go to your GitHub repo → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `RENDER_DEPLOY_HOOK_URL`
   - Value: (paste the deploy hook URL)

3. **Push to trigger deploy:**
   ```bash
   git push origin main
   ```

---

## Important Notes

### ⚠️ Free Tier Limitations

- Render's free tier spins down after 15 minutes of inactivity
- First request after spin-down takes ~30-60 seconds (cold start)
- For production, consider upgrading to a paid plan ($7/month)

### 📁 File Persistence

- Free tier has **no persistent disk**
- FAISS index is rebuilt on every deployment
- For persistent storage, upgrade to a paid plan with disk

### 🔐 Environment Variables

- Never commit API keys to GitHub
- Always use Render's Environment Variables feature
- You can delete the old VM secrets from GitHub

### 🌐 Custom Domain (Optional)

- Free plan: `your-app.onrender.com`
- Paid plan: Add custom domain in Render settings

---

## Troubleshooting

### Build Fails

- Check that `requirements.txt` is in root directory
- Verify Python version compatibility (Render uses Python 3.11 by default)

### App Crashes on Startup

- Check logs in Render dashboard
- Verify all environment variables are set
- Ensure `knowledge.txt` exists in repo

### "Service Unavailable" Error

- App might be initializing (takes ~30 seconds on first start)
- Check `/health` endpoint to see initialization status
- Free tier: might be spinning up from sleep

### Out of Memory

- Free tier has 512MB RAM limit
- If your FAISS index is too large, consider:
  - Reducing chunk size
  - Using fewer documents
  - Upgrading to paid plan (more RAM)

---

## Migration Checklist

- [x] Created `render.yaml` configuration
- [x] Created `Procfile` (backup)
- [x] Updated `.github/workflows/deploy.yml`
- [ ] Push code to GitHub
- [ ] Create Render account
- [ ] Deploy using Blueprint
- [ ] Set environment variables in Render
- [ ] Test deployment
- [ ] Update frontend URL (if applicable)
- [ ] Delete old VM secrets from GitHub (optional)

---

## Next Steps After Deployment

1. **Test Your API:**

   ```bash
   # Replace with your Render URL
   export TRIX_URL="https://trix-chatbot.onrender.com"

   curl $TRIX_URL/health
   curl -X POST $TRIX_URL/initialize
   curl -X POST $TRIX_URL/ask -H "Content-Type: application/json" \
     -d '{"question": "What is Trikon 2025?"}'
   ```

2. **Monitor Logs:**
   - Render Dashboard → Your Service → Logs tab

3. **Set Up Alerts:**
   - Render Dashboard → Your Service → Settings → Notifications

4. **Update Your Frontend:**
   - Change API base URL from old VM to new Render URL

---

## Render vs Azure VM Comparison

| Feature         | Azure VM (Old)          | Render (New)                |
| --------------- | ----------------------- | --------------------------- |
| **Cost**        | ~$35-50/month           | **Free** (with limitations) |
| **Setup**       | Manual (SSH, PM2, etc.) | **Automatic**               |
| **Auto-deploy** | Via GitHub Actions      | **Built-in**                |
| **Scaling**     | Manual                  | **Automatic**               |
| **SSL/HTTPS**   | Manual setup            | **Automatic**               |
| **Monitoring**  | Manual setup            | **Built-in**                |
| **Persistence** | Full disk               | None (free tier)            |
| **Always On**   | Yes                     | No (free tier sleeps)       |

---

**Need help?** Check [Render's Python documentation](https://render.com/docs/deploy-flask)
