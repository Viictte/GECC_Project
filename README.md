# GECC_Project

This project involves the analysis of datasets crawled from anonymous chatting platforms such as Goop, Lihkg. The data was collected from college students in Hong Kong.

## Data Processing and Cleaning

The raw data was processed and cleaned using Python. The cleaning process involved removing unnecessary characters, handling missing values, and transforming the data into a suitable format for analysis.

## Exploratory Data Analysis (EDA)

Detailed Exploratory Data Analysis (EDA) was performed to uncover the informative patterns in these datasets. The EDA involved the following steps:

1. **Data Overview:** The data was first loaded into a pandas DataFrame to get an overview of the data structure, the types of data, and the missing values.

2. **Sentiment Analysis:** Sentiment analysis was performed on the post titles to classify the sentiments as positive, negative, or neutral. This provided insights into the overall sentiment of the posts on the platforms.

3. **Topic Modeling:** Latent Dirichlet Allocation (LDA) was used for topic modeling. This helped in identifying the main topics that the posts were about. The topics were then visualized using a word cloud.

4. **Type-Token Ratio (TTR):** The TTR was calculated for each post to measure the lexical diversity. A higher TTR indicates a wider range of vocabulary, suggesting that users are expressing a wide range of ideas and topics in their posts.

5. **Keyword Analysis:** Keyword analysis was performed to identify the most common words and phrases in the post titles. This helped in understanding the most discussed topics on the platforms.

The findings from the EDA provide valuable insights into the behavior and preferences of the users on these platforms, which can be useful for various applications such as improving user experience, content moderation, and understanding user trends.
