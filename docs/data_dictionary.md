# **Data Dictionary**

## **Introduction**

This document serves as a comprehensive guide to understanding the datasets used in the Dynamic Topic Modeling project. The data dictionary provides detailed descriptions of each variable present in the raw and processed datasets, including data types, example values, and any preprocessing steps that were applied.

## **Dataset Overview**

### **Source of Data**

The data for this project was collected from Quantinar, an online platform offering various courses. The dataset includes information about courses, such as titles, descriptions, instructors, and dates of updates.

### **Structure of Datasets**

- **Raw Data**: The original data collected from Quantinar, containing all the fields as they appear in the platform's interface.
- **Processed Data**: The dataset after undergoing preprocessing steps such as tokenization, lemmatization, and the removal of stopwords. This processed dataset is used for topic modeling and other analyses.

## **Column Descriptions**

### **Raw Data (`quantinar_courses_raw.json`)**

| **Column Name**       | **Data Type** | **Description**                                                                                                | **Example Values**                                    |
|-----------------------|---------------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `title`               | String        | The title of the course.                                                                                         | "Cryptocurrency Dynamics"                             |
| `url`                 | String        | The URL where the course can be accessed.                                                                        | "https://quantinar.com/course/29/___"                 |
| `description`         | Text          | A brief description of the course content.                                                                       | "This video gives insights on the dynamics of the cryptocurrency sector." |
| `instructors`         | String        | The name(s) of the instructor(s) who created the course.                                                         | "Konstantin Haeusler"                                 |
| `last_updated`        | Date          | The date when the course content was last updated.                                                               | "7th November 2022"                                   |
| `processed_text`      | List          | A list of processed tokens from the course description, used in text analysis.                                   | `['cryptocurrency', 'dynamics', ...]`                 |
| `metadata`            | JSON          | Additional metadata associated with the course, such as page status, source URL, and more.                       | `{"pageStatusCode": 200, "sourceURL": "https://..."} `|
| `linksOnPage`         | List          | A list of all URLs present on the course page.                                                                   | `["https://quantinar.com", "https://quantlet.com"]`   |

### **Processed Data (`preprocessed_courses.csv`)**

| **Column Name**       | **Data Type** | **Description**                                                                                                | **Example Values**                                    |
|-----------------------|---------------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `title`               | String        | The title of the course.                                                                                         | "Cryptocurrency Dynamics"                             |
| `url`                 | String        | The URL where the course can be accessed.                                                                        | "https://quantinar.com/course/29/___"                 |
| `description`         | Text          | A brief description of the course content.                                                                       | "This video gives insights on the dynamics of the cryptocurrency sector." |
| `instructors`         | String        | The name(s) of the instructor(s) who created the course.                                                         | "Konstantin Haeusler"                                 |
| `last_updated`        | Date          | The date when the course content was last updated.                                                               | "7th November 2022"                                   |
| `processed_text`      | List          | A list of processed tokens from the course description, used in text analysis.                                   | `['cryptocurrency', 'dynamics', ...]`                 |
| `month`               | Period        | The month derived from `last_updated`, used to create time slices for topic modeling.                            | "2022-11"                                             |

## **Data Preprocessing**

### **Steps Applied**

1. **Tokenization**: The course descriptions were broken down into individual words or tokens.
2. **Lowercasing**: All tokens were converted to lowercase to ensure uniformity.
3. **Stopword Removal**: Common English words that do not contribute to the meaning (e.g., "and", "the") were removed.
4. **Lemmatization**: Tokens were reduced to their base or root form (e.g., "running" to "run").
5. **Date Conversion**: The `last_updated` field was converted to a datetime format to facilitate time-based analysis.

### **Example of Preprocessing**

- **Original Description**: "This video gives insights on the dynamics of the cryptocurrency sector."
- **Processed Text**: `['video', 'give', 'insight', 'dynamic', 'cryptocurrency', 'sector']`

## **Conclusion**

This data dictionary provides a clear understanding of the datasets used in the project. Each column is meticulously documented to ensure that anyone working with the data can do so with full comprehension of its contents and the preprocessing steps involved.

