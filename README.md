# 🫀 Healthcare Edge Computing: Priority-Based Task Scheduling

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Java](https://img.shields.io/badge/Java-ED8B00?style=flat&logo=openjdk&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![iFogSim2](https://img.shields.io/badge/iFogSim2-simulation-4B8BBE?style=flat)
![EdgeSimPy](https://img.shields.io/badge/EdgeSimPy-simulation-2E8B57?style=flat)

A priority-based task scheduling and resource allocation system for **healthcare IoT** using **edge computing**. It implements the Priority-based Task Scheduling (PTS) mechanism for smart-hospital environments.

> Capstone project, Team 24 (Edge Computing course, 2025-26)

## Overview

IoT sensors stream patient vitals in real time. Each task gets an urgency level (k-value) based on medical emergency criteria, and the scheduler decides whether to process it on a nearby **edge device** (low latency, critical cases) or offload it to the **cloud**.

## Key features

- **Real-time health monitoring** of heart rate, blood pressure and glucose levels
- **Priority system** that computes urgency (k-values) from medical emergency criteria
- **Edge vs cloud routing** of tasks based on urgency
- **ML risk prediction** with a RandomForest model for patient-specific priority
- **Real medical data** from the UCI Heart Disease dataset (200+ patient records)

## Repository layout

| Folder | Contents |
| --- | --- |
| `Team24_HeartAttack/` | Main Python implementation (config, data, models, simulation) |
| `Code_EdgeSimPy/` | Simulation built on EdgeSimPy |
| `Code_iFogSim2/` | Java simulation on iFogSim2 (sensors, actuators, priority manager, risk prediction) |
| `Documentation/` | Reports, review slides and the research paper |

## Getting started

```bash
git clone https://github.com/suhas107/capstone-project-24_heartattack.git
cd capstone-project-24_heartattack/Team24_HeartAttack
python -m venv venv
venv\Scripts\activate        # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python main.py
```

## Documentation

See the [`Documentation`](Documentation/) folder for the project reports, review presentations and the research paper.
