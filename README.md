# Heart Disease Prediction Using Big Data, Data Warehousing, and Machine Learning

This project was developed as a graduation project with the goal of building an intelligent system that can predict the likelihood of heart disease using modern data engineering and machine learning techniques.

The repository combines several important areas of data science and analytics, including:
- data collection and preprocessing
- data augmentation
- big data processing
- data warehousing
- machine learning model development
- business intelligence and reporting

The main idea is simple but powerful: use real healthcare data to build a predictive model that can help identify patients at risk of heart disease earlier and more accurately.

---

## Project Overview

Heart disease remains one of the leading causes of death worldwide. Early detection can significantly improve patient outcomes and reduce the burden on healthcare systems. This project aims to support that goal by creating a data-driven prediction system that uses historical patient data and machine learning models to estimate the risk of heart disease.

This repository demonstrates how different technologies can work together in a realistic data pipeline:
- data is collected and processed
- cleaned and transformed for analysis
- stored in a warehouse structure
- used to train predictive models
- visualized through reports and dashboards

The project is not just a notebook experiment; it reflects a full end-to-end analytics workflow that resembles how real-world data-driven systems are built in industry.

---

## Why This Project Matters

Healthcare decisions are often based on a combination of clinical signs, patient history, and risk factors. By analyzing patient data with machine learning, we can:
- identify high-risk individuals earlier
- reduce diagnosis delays
- support clinicians with better decision-making tools
- improve the quality of healthcare analysis

This project focuses on turning raw medical data into something useful, actionable, and predictive.

---

## Main Objectives

The project was designed to achieve the following goals:

1. Build a robust data warehouse for storing healthcare and heart disease-related data.
2. Prepare and clean a comprehensive heart disease dataset.
3. Apply big data tools and processing frameworks to handle large-scale data.
4. Use machine learning algorithms to predict heart disease risk.
5. Explore data augmentation techniques to improve model training.
6. Integrate stream processing concepts using Kafka.
7. Design a reporting and dashboarding layer for business and stakeholder insights.
8. Compare multiple modeling approaches and evaluate their performance.

---

## Technologies and Tools Used

This project brings together multiple technologies from data engineering, analytics, and machine learning.

| Area | Technology |
|------|-----------|
| Data Processing | Python, Jupyter Notebook, Apache Spark |
| Streaming | Apache Kafka |
| Databases | MongoDB |
| Data Warehousing | PostgreSQL |
| Machine Learning | scikit-learn, TensorFlow/Keras |
| Data Augmentation | GAN-based techniques |
| Visualization | Power BI |
| Data Formats | CSV, JSON |
| Environment | Jupyter Notebooks |

This repository includes both data engineering and model-building artifacts, which makes it a strong example of a practical end-to-end analytics project.

---

## Project Structure

The repository contains several folders and files representing different phases of the project:

- `Graduation_Project/Model.ipynb`  
  Main notebook for model development and training

- `Graduation_Project/Prediction.ipynb`  
  Notebook used for making predictions using the trained model

- `Graduation_Project/Pykafka.ipynb`  
  Kafka-related notebook for data streaming and ingestion flow

- `Graduation_Project/augumentaion/`  
  Contains augmentation activities and generated datasets

- `Graduation_Project/random_forest_model/`  
  Model-related artifacts and saved outputs

- `Graduation_Project/DW_ERD.pgerd`  
  Data warehouse ERD file

- `Graduation_Project/DW_Tables.txt`  
  Data warehouse table definitions and schema description

- `Graduation_Project/GP.pbix`  
  Power BI dashboard project

- `Graduation_Project/GAN.png`  
  Visual representation of the GAN model used for data augmentation

- `Graduation_Project/all_data.csv`  
  Combined dataset used throughout project experiments

- `Graduation_Project/Data_Description.docx`  
  Dataset documentation

- `Graduation_Project/Heart Disease Dataset (Comprehensive) _ IEEE DataPort.html`  
  Reference to the dataset source

---

## Data Pipeline

The project follows a realistic data pipeline workflow:

1. Data acquisition  
   The heart disease dataset is collected and reviewed.

2. Data understanding  
   The dataset is analyzed to understand its structure, features, and health-related relevance.

3. Data preprocessing  
   Missing values, inconsistent data, and irrelevant fields are handled before modeling.

4. Data augmentation  
   Generative techniques are explored to enrich the dataset and support model training.

5. Big data processing  
   Spark and other distributed tools are used to process large-scale data.

6. Streaming integration  
   Kafka is used to simulate real-time or near-real-time data ingestion.

7. Data warehouse design  
   Structured data is organized in a warehouse for analytical access.

8. Modeling  
   Machine learning algorithms such as Random Forest are trained and evaluated.

9. Visualization  
   Power BI reports are created to help stakeholders interpret the results.

---

## Machine Learning Approach

The project explores predictive analytics for heart disease risk by using patient health features as inputs and classification models to predict whether a patient is likely to have heart disease.

The model development process includes:
- data cleaning
- feature inspection
- train/test splitting
- model training
- performance evaluation
- improvement through tuning and comparison

The repository also includes a Random Forest model and experimentation files, showing the importance of comparing models and selecting the most suitable predictive algorithm.

---

## Data Warehousing and Analytics Layer

One of the distinctive parts of this project is the data warehousing component. Instead of working only with raw notebooks, the project also includes:
- warehouse schema design
- entity relationship modeling
- table descriptions
- analytical structure for reporting

This is useful because it demonstrates how medical or business data can be organized in a structured way for better querying, long-term use, and dashboarding.

---

## Dashboard and Reporting

A Power BI file is included in the project to visualize findings and operational insights. This allows users to translate raw predictions into meaningful business or healthcare reporting.

Power BI can help answer questions such as:
- Which patient attributes are most associated with heart disease?
- Which model performs best?
- How does the risk profile vary across samples?
- What patterns can support medical decision-making?

---

## Key Project Highlights

- Predictive healthcare analytics project
- Real-world data-driven dataset
- Integration of big data tools
- Data warehouse design and schema planning
- Machine learning for classification
- Data augmentation with GAN-inspired methods
- Dashboarding and reporting support
- End-to-end project structure suitable for academic presentation

---

## How to Use This Repository

1. Open the repository in GitHub or clone it locally.
2. Review the dataset and documentation files.
3. Explore the notebooks in order:
   - `Model.ipynb`
   - `Prediction.ipynb`
   - `Pykafka.ipynb`
4. Check the augmentation folder for preprocessing and enrichment experiments.
5. Review the warehouse files and diagram for the storage design.
6. Open the Power BI project to explore reporting visuals.

---

## Suggested Workflow

A typical execution flow for this project is:

- Prepare and clean the dataset
- Run data augmentation if needed
- Train and evaluate the machine learning model
- Save model outputs
- Integrate Kafka-based streaming
- Use warehouse schema for organization
- Visualize results in Power BI

---

## Project Outcome

The final goal of the project is to create a practical solution that combines:
- big data processing
- data warehousing
- machine learning
- reporting and analytics

This makes the project more than just a technical exercise — it reflects how modern data systems are used to solve real-world problems in healthcare and decision support.

---

## Conclusion

This project demonstrates the power of combining data engineering and artificial intelligence to solve a meaningful problem in the healthcare domain. By integrating machine learning with big data and warehouse concepts, it builds a strong foundation for intelligent decision support systems.

The repository is a valuable example of how academic projects can connect theory to practical implementation using modern tools and technologies.

---

## License

This project is intended for academic and educational use. Please review the repository files and associated data sources before reusing them in other contexts.

---

## Acknowledgment

This project was built to explore how modern data and machine learning techniques can support heart disease prediction and improve healthcare decision-making.
