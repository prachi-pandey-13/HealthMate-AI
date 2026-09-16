# 💙 HealthMate AI

### AI-Powered Preventive Health & Wellness Assistant

HealthMate AI is an AI-powered preventive health and wellness assistant designed to provide accessible, simple, and responsible educational information about everyday health and well-being.

The project was developed as part of the **IBM SkillsBuild AI for Sustainability Virtual Internship** and is aligned with **UN Sustainable Development Goal 3 (SDG 3) — Good Health and Well-being**.

---

## 🎯 Problem Statement

How might we use AI to provide accessible and responsible preventive-health and wellness information so that individuals can make better-informed decisions about their well-being?

---

## 💡 Solution

HealthMate AI combines:

- 🤖 IBM Granite through IBM watsonx.ai
- 📚 Retrieval-Augmented Generation (RAG)
- 🔎 FAISS similarity search
- 🧠 Sentence Transformers
- 🌱 Preventive health knowledge
- 🛡️ Responsible AI and safety checks
- 💻 Streamlit user interface

Users can ask general questions about sleep, nutrition, physical activity, mental well-being, and preventive health.

---

## ✨ Key Features

- 💬 Conversational health and wellness assistance
- 😴 Sleep and rest guidance
- 🥗 Healthy nutrition information
- 🏃 Physical activity guidance
- 🧠 Mental well-being information
- 🩺 General preventive health guidance
- 📚 Knowledge-base retrieval using RAG
- 🚨 Basic emergency-situation safety detection
- 🛡️ Responsible AI safeguards
- 🔎 Knowledge transparency showing retrieved information
- 🔐 API key protection using environment variables

---

## 🏗️ System Architecture

```text
User
  ↓
Streamlit Interface
  ↓
Safety Check
  ↓
RAG Retrieval
  ↓
Health Knowledge Base
  ↓
FAISS + Sentence Transformers
  ↓
Relevant Knowledge
  ↓
IBM Granite 4 H Small
  ↓
Responsible AI Prompt
  ↓
Final Response

## 🖥️ Application Preview

### Home Page

![HealthMate AI Home](screenshots/home.png)

### AI Response

![HealthMate AI Response](screenshots/response.png)