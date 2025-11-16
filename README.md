 Smart Waste Classifier

A small service and web UI for classifying waste (plastic, paper, organic, etc.) using a local ML model.

## Features
- Upload images and get predicted waste category
- Local ONNX model support
- Simple web UI with Tailwind CSS
- Configurable via `.env`

## Prerequisites
- Node.js 16+ (or compatible)
- npm, yarn or pnpm
- Git (optional)
- Model file (ONNX) placed at the path configured in `.env`

## Quick start (PowerShell)
1. Install dependencies
````powershell
npm install
# or
# yarn
# pnpm install
