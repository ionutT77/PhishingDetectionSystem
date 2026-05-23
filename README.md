# AI Phishing Detection System

https://aiphishingdetectionsystem.streamlit.app/

An interactive machine learning application built with Streamlit and Neural Networks to detect phishing URLs with 95% accuracy.

### The Build & The Data

A major focus of this project was the data itself. Rather than relying on standard, pre-existing sources, I wanted to build the foundation from the ground up. I independently designed the data architecture and selection criteria to create a proprietary dataset of over 2 million URL entries. Managing the complex data handling and feature extraction at this scale was a heavy lift, but it gave me complete control over the quality of the data feeding the model.

### The Solution

Using that custom dataset, I trained a Neural Network to accurately flag malicious links. The resulting system serves as a highly effective automated tool for alert triage and threat intelligence correlation, helping to quickly filter out real phishing threats. To bring the model out of the backend and make it actively usable, I deployed it as an interactive web application using Streamlit.

---

**Project Type:** Machine Learning / Cybersecurity Tool

**Core Tech:** Neural Networks, Streamlit, Custom Data Architecture

**Scale:** 2 Million+ Proprietary URL Entries

**Performance:** 95% Detection Accuracy
