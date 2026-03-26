module.exports = {
  apps: [
    {
      name: "trix-chatbot",
      script: "server2.py",
      interpreter: "./venv/bin/python3",
      cwd: "/home/azureuser/trikon-2",  // Update this to your actual deployment path
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "500M",
      env: {
        NODE_ENV: "production",
      },
      error_file: "./logs/err.log",
      out_file: "./logs/out.log",
      log_file: "./logs/combined.log",
      time: true,
    },
  ],
};
