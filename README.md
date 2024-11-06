### Application description:
The aim of the project was to create a universal application that allows for detecting the most important features in a given data set. In short - the user uploads data or loads a ready data set in the appropriate format, then selects automatic detection of the column they want to analyze or makes this selection themselves. Finally, they receive a generated graph of the significance of features that have the greatest impact on the previously selected column. The user also receives a clear description of the graph along with recommendations - what can be improved to, for example, improve the analyzed data.

### Main functionalities:
* the user can load a CSV/JSON file with data or use a ready-made sample dataset,
* LLM model recognizes column names and gives them appropriate descriptions,
* the user indicates the target column -> additionally, they can use automatic column detection (generated by LLM),
* the application automatically recognizes whether the loaded data is related to the regression or classification problem and selects the appropriate AI model training algorithm on this basis,
* based on the trained model, a chart containing the most important features is displayed,
* finally, the user receives a clear description of the chart along with recommendations - what actions to implement to improve the results related to the analyzed target data column.

### Dependencies:
* streamlit,
* pycaret,
* pandas,
* mathplotlib,
* python-dotenv,
* langfuse,
* instructor,
* pydantic,
* boto3.
