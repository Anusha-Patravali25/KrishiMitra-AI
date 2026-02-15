# 🌾 KrishiMitra-AI

## AI-Powered Multilingual Crop Advisory Platform for Bharat

KrishiMitra-AI is an intelligent agricultural advisory platform built using AWS Serverless Architecture, Amazon Bedrock, and Retrieval-Augmented Generation (RAG) to empower smallholder farmers across India.

It delivers AI-driven crop disease diagnosis, region-specific treatment guidance, multilingual accessibility, and economic impact estimation in a scalable and cost-efficient manner.

---

## 🚜 Problem Statement

Smallholder farmers in India face:

1. Difficulty identifying crop diseases early
2. Limited access to agricultural experts
3. Language barriers in advisory systems
4. Financial uncertainty due to untreated crop damage
5. Lack of region-specific agricultural guidance

KrishiMitra-AI addresses these challenges using AI-powered automation on AWS.

---

## 🎯 Solution Overview

KrishiMitra-AI enables farmers to:

1. Upload crop images for AI-based disease diagnosis
2. Describe crop symptoms in regional languages
3. Receive region-specific agricultural recommendations
4. Understand financial impact of diseases
5. Access optional voice-based advisory output

All powered by Amazon Bedrock with a Retrieval-Augmented Generation framework.

---

## 🏗️ Architecture Overview

Core AWS services used:

1. Amazon S3 for secure image and audio storage
2. AWS Lambda for serverless compute
3. Amazon API Gateway for REST API management
4. Amazon Rekognition for crop image analysis
5. Amazon Bedrock for AI-powered diagnosis generation
6. Vector Database for semantic knowledge retrieval
7. AWS IAM for secure access control
8. Amazon CloudWatch for monitoring and logging

---

## 🔎 Key Features

### 1. Image-Based Crop Disease Detection

1. Accepts JPEG, PNG, and HEIC formats
2. Detects anomalies using Rekognition
3. Generates contextual diagnosis via Bedrock

### 2. Text-Based Symptom Analysis

1. Accepts multilingual symptom descriptions
2. Uses semantic embeddings for similarity search
3. Supports low-connectivity usage scenarios

### 3. Region-Specific RAG Knowledge Retrieval

1. Filters responses by crop type
2. Filters by geographic region
3. Retrieves top relevant agricultural documents before LLM generation

### 4. Treatment and Prevention Advisory

1. Provides organic and chemical treatment options
2. Includes dosage and application guidance
3. Prioritizes cost-effective local solutions

### 5. Economic Impact Estimation

1. Estimates yield loss percentage
2. Calculates potential financial loss
3. Provides treatment cost comparison

### 6. Multilingual and Voice Support

Supported languages:

1. Hindi
2. Tamil
3. Telugu
4. Kannada
5. Bengali
6. Marathi

Optional advisory voice output is generated through a TTS engine.

---

## 📊 Success Metrics

1. Diagnostic accuracy above 85 percent
2. Response time under 5 seconds
3. Cost per query below ₹2
4. Platform availability above 99.5 percent
5. User satisfaction score above 4.0 out of 5

---

## 🔐 Security and Privacy

1. AES-256 encryption for stored data
2. TLS 1.2 or higher for data in transit
3. IAM least-privilege access model
4. No personal data stored without consent
5. User-requested data deletion supported

---

## 🏆 Hackathon Submission

Submitted for AWS AI for Bharat Hackathon 2026.

Demonstrates practical implementation of:

1. Amazon Bedrock
2. Retrieval-Augmented Generation architecture
3. Scalable AWS serverless infrastructure
4. Multilingual AI accessibility

## 👩‍💻 Author

Anusha Patravali

GitHub: https://github.com/Anusha-Patravali25

✨ Empowering Bharat’s farmers through accessible AI.
