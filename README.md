# Machine Learning-Based Predictive Analytics for Supply Chain Optimization

## Team Members

| Name | ID Number |
|---|---|
| Shiva Sai Vinesh | 2420030691 |
| Harsha Vardhan Reddy | 2420030748 |
| Venkatesh | 2420030134 |
| Shashank Reddy | 2420030034 |

## Supervisor

Dr.K.Swapnika

## Abstract

Modern supply chains generate large and complex datasets involving demand signals, inventory levels, supplier performance, transportation conditions, and external factors such as weather and market trends. Traditional supply chain management relies on static rules and historical averages, which struggle to adapt to real-time demand shifts, supplier disruptions, and transportation delays.

Therefore, there is a need for a scalable, data-driven predictive analytics framework that can forecast supply chain risks and demand patterns while supporting proactive, informed decision-making.

This project proposes a Unified Predictive Analytics Framework that integrates historical transactional, inventory, supplier, and logistics data using Machine Learning to enable demand forecasting, inventory optimization, and supplier risk prediction. The system aims to provide accurate, data-driven insights that support supply chain planning, resilience, and operational efficiency.

## Problem Statement

Supply chains generate large and complex datasets involving demand, inventory levels, supplier performance, transportation times, and pricing.

Existing supply chain management approaches often rely on static, rule-based methods that cannot adapt to real-time disruptions or fast-changing demand patterns, resulting in stockouts, excess inventory, and delayed responses to risk.

Therefore, there is a need for a scalable Machine Learning-based framework that can predict supply chain risks and demand fluctuations while enabling proactive, data-driven decision-making.

## Project Objectives

* To develop machine learning models for forecasting demand and predicting supply chain disruptions using historical business data.
* To identify and visualize important supply chain trends, patterns, and risk factors through comprehensive data analysis.
* To use predictive analytics techniques to optimize inventory levels and identify supplier and logistics risk.
* To develop a Unified Predictive Analytics Framework that integrates demand, inventory, supplier, and transportation data using Machine Learning.
* To provide clear, data-driven insights that support supply chain planning and proactive decision-making.

## Proposed Work and Uniqueness

The proposed system:

* Combines demand forecasting, inventory optimization, and supplier risk analysis in one system.
* Uses multiple supply chain factors instead of focusing on a single function in isolation.
* Integrates Machine Learning models across forecasting, classification, and optimization tasks.
* Provides comparative evaluation of models to identify which factors and techniques drive the most accurate predictions.
* Aims to provide early-warning insights for real-world supply chain decision-making.
* Integrates multiple supply chain data sources through a Unified Predictive Analytics Framework.

## System Architecture

```
Supply Chain Data Sources
              |
              v
      Data Collection
              |
              v
    Data Cleaning & Integration
              |
              v
      Data Processing
              |
              v
   Exploratory Data Analysis
              |
              v
      ML / DL Models
              |
              v
   Demand & Risk Prediction
              |
              v
     Risk Classification
      Low / Medium / High
              |
              v
   Feature Importance Analysis
              |
              v
  Supply Chain Insights Dashboard
```

## Benchmark Datasets

The project uses a multi-source approach to collect supply chain information.

**Dataset 1 – Online Retail / Transactional Dataset**

Purpose:
* Order history
* SKU-level demand
* Sales trends

**Dataset 2 – Inventory and Warehouse Data**

Contains:
* Stock levels
* Reorder points
* Lead times

**Dataset 3 – Supplier and Logistics Data**

Contains:
* Supplier delivery performance
* Transportation times
* Delay records

## Dataset Integration

The collected datasets will be cleaned, integrated, preprocessed, and analysed to identify supply chain patterns and support demand and risk prediction.

```
Transactional Data
       +
Inventory Data
       +
Supplier & Logistics Data
       |
       v
Unified Supply Chain Dataset
       |
       v
Data Cleaning & Integration
       |
       v
Demand & Risk Analysis
```

## Technologies and Tools

**Programming**
* Python

**Data Processing**
* Pandas
* NumPy

**Machine Learning**
* Scikit-learn
* Random Forest
* XGBoost
* LightGBM
* CatBoost

**Deep Learning**
* LSTM
* Other suitable deep-learning models when required

**Visualization and Dashboard**
* Streamlit
* Power BI
* Python visualization libraries

**Development Environment**
* Local system
* Google Colab

## Project Plan

**Phase 1 – Literature Review**
Study 10–15 recent research papers. Identify limitations and research gaps.
Status: Completed / Review-1

**Phase 2 – Dataset Collection**
Collect demand, inventory, and supplier datasets. Perform data cleaning and integration.
Status: Next Phase

**Phase 3 – Data Processing**
Perform preprocessing and exploratory analysis.
Status: Planned

**Phase 4 – Model Development**
Train multiple ML/DL models. Compare prediction performance.
Status: Planned

**Phase 5 – Risk & Inventory Analysis**
Develop risk categories such as Low, Medium, High. Analyze factors responsible for each risk.
Status: Planned

**Phase 6 – Visualization**
Build a dashboard showing predictions, trends, and risk levels.
Status: Planned

**Phase 7 – Evaluation & Documentation**
Compare models. Validate results. Document findings and prepare research paper.
Status: Planned

## Project Phase Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Literature Review | Completed / Review-1 |
| Phase 2 | Dataset Collection | Next Phase |
| Phase 3 | Data Processing | Planned |
| Phase 4 | Model Development | Planned |
| Phase 5 | Risk & Inventory Analysis | Planned |
| Phase 6 | Visualization | Planned |
| Phase 7 | Evaluation & Documentation | Planned |

## Research Gap

The project addresses the following research gaps:

* Many supply chain prediction models focus primarily on accuracy rather than practical decision support.
* Several ML-based supply chain models are evaluated in isolation rather than across integrated functions (demand, inventory, supplier risk).
* Existing studies often focus on one supply chain function rather than combining multiple factors.
* Large-scale heterogeneous supply chain datasets create challenges in data integration and quality.
* Limited research combines demand forecasting, inventory optimization, and supplier risk prediction in a single framework.
* Many studies evaluate models technically but provide limited guidance on which factors drive predicted risk or demand shifts.

## Risk Analysis

The system will classify supply chain risks into categories such as:

* Low Risk
* Medium Risk
* High Risk

The system will also analyse the factors responsible for each predicted risk level, such as supplier delay history, demand volatility, and transportation conditions.

## Dashboard

The proposed interactive dashboard will display:

* Demand trends
* Inventory levels
* Risk predictions
* Risk levels
* Important contributing factors
* Supply chain patterns
* Analytical insights

The dashboard may be implemented using Streamlit or Power BI.

## Setup and Execution

**1. Clone the Repository**

```
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

**2. Create a Virtual Environment**

```
python -m venv venv
```

**3. Activate the Virtual Environment**

For Windows:
```
venv\Scripts\activate
```

For Linux/macOS:
```
source venv/bin/activate
```

**4. Install Dependencies**

If a requirements.txt file is available:
```
pip install -r requirements.txt
```

Otherwise, install the major project dependencies:
```
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib streamlit
```

**5. Add Datasets**

Place the downloaded datasets inside the appropriate directories:

```
data/
├── transactional/
├── inventory/
├── supplier_logistics/
└── processed/
```

**6. Perform Preprocessing**

```
python <preprocessing_script>.py
```

**7. Run the Machine Learning Models**

```
python <model_script>.py
```

**8. Run the Dashboard**

For Streamlit:
```
streamlit run app.py
```

If Power BI is used, load the processed dataset into Power BI and open the dashboard/report.

## Repository Structure

```
project-root/
│
├── README.md
├── data/
│   ├── transactional/
│   │   ├── orders/
│   │   ├── demand/
│   │   └── sales/
│   │
│   ├── inventory/
│   │   ├── stock_levels/
│   │   ├── reorder_points/
│   │   └── lead_times/
│   │
│   ├── supplier_logistics/
│   │   ├── delivery_performance/
│   │   ├── transportation/
│   │   └── delays/
│   │
│   └── processed/
│
├── notebooks/
│   ├── data_analysis/
│   ├── eda/
│   └── experiments/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   └── risk_analysis/
│
├── dashboard/
├── results/
├── docs/
├── requirements.txt
└── app.py
```

## Literature Review

The literature review focuses on the following areas:

* Demand Forecasting using AI/ML
* Big Data for Supply Chain Analysis
* Inventory Optimization Techniques
* Supplier Risk Prediction
* Transportation and Delivery Time Prediction
* Supply Chain Time-Series Forecasting
* IoT and Predictive Maintenance for Logistics

The literature review identifies the limitation that many existing approaches focus on a single supply chain function while providing limited integration of demand forecasting, inventory optimization, and supplier risk assessment.

## Current Project Status

**Current Phase:** Phase 1 – Literature Review / Review-1

**Completed**
* Problem statement
* Project objectives
* Initial literature review
* Research gap identification
* Proposed work and uniqueness
* Initial system architecture
* Benchmark dataset identification
* Project phase planning

**Next Step**

Phase 2 – Dataset Collection

The next stage will focus on collecting transactional, inventory, and supplier/logistics datasets, followed by data cleaning and integration.

## License

This project is developed for academic and research purposes.
