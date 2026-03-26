#!/bin/bash

# ============================================
# VM Initial Setup Script for Trix Chatbot
# ============================================
# Run this ONCE on your Azure VM after cloning the repo

set -e  # Exit on any error

echo "🚀 Starting VM setup for Trix Chatbot..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running on VM
if [ ! -d "/home/azureuser" ]; then
    echo -e "${RED}Warning: This script is designed for Azure VMs with user 'azureuser'${NC}"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# 1. Install Node.js and npm if not installed
if ! command -v node &> /dev/null; then
    echo -e "${BLUE}📦 Installing Node.js and npm...${NC}"
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
else
    echo -e "${GREEN}✅ Node.js already installed$(node -v)${NC}"
fi

# 2. Install PM2
if ! command -v pm2 &> /dev/null; then
    echo -e "${BLUE}📦 Installing PM2...${NC}"
    sudo npm install -g pm2
else
    echo -e "${GREEN}✅ PM2 already installed$(pm2 -v)${NC}"
fi

# 3. Create virtual environment
if [ ! -d "venv" ]; then
    echo -e "${BLUE}🐍 Creating Python virtual environment...${NC}"
    python3 -m venv venv
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

# 4. Activate venv and install dependencies
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Create logs directory
echo -e "${BLUE}📁 Creating logs directory...${NC}"
mkdir -p logs

# 6. Update ecosystem.config.js with current directory
CURRENT_DIR=$(pwd)
echo -e "${BLUE}📝 Updating ecosystem.config.js with path: ${CURRENT_DIR}${NC}"

# Use sed to replace the path in ecosystem.config.js
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s|cwd: \".*\"|cwd: \"${CURRENT_DIR}\"|g" ecosystem.config.js
else
    # Linux
    sed -i "s|cwd: \".*\"|cwd: \"${CURRENT_DIR}\"|g" ecosystem.config.js
fi

# 7. Create a sample .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${BLUE}📝 Creating .env file from example...${NC}"
    cp .env.example .env
    echo -e "${RED}⚠️  IMPORTANT: Edit .env and add your GOOGLE_API_KEY!${NC}"
fi

# 8. Setup PM2 startup
echo -e "${BLUE}⚙️  Setting up PM2 to start on boot...${NC}"
pm2 startup | tail -n 1 > /tmp/pm2_startup_cmd.sh
sudo bash /tmp/pm2_startup_cmd.sh
rm /tmp/pm2_startup_cmd.sh

# 9. Start the application
echo -e "${BLUE}🚀 Starting application with PM2...${NC}"
pm2 start ecosystem.config.js
pm2 save

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "Next steps:"
echo -e "1. Edit .env file: ${BLUE}nano .env${NC}"
echo -e "2. Add your GOOGLE_API_KEY"
echo -e "3. Restart PM2: ${BLUE}pm2 restart trix-chatbot${NC}"
echo -e "4. Check logs: ${BLUE}pm2 logs trix-chatbot${NC}"
echo -e "5. Check status: ${BLUE}pm2 status${NC}"
echo ""
echo -e "Your app will be running on: ${GREEN}http://localhost:3000${NC}"
echo ""
