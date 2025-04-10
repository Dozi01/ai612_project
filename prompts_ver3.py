answerable_classification_system_prompt = """You are an AI assistant tasked with determining whether a given question can be answered using the provided SQL tables. Your job is to classify the question as answerable or unanswerable and provide a score indicating how answerable it is.

### Examine the SQL tables:
{SQL_TABLES}

### Here is the assumptions:
{SQL_ASSUMPTIONS}

"""

answerable_classification_user_prompt = """Your task is to classify the following user question as answerable or unanswerable based on the SQL tables and the assumptions:

### [Unanswerable questions]
- Vague questions
- Questions that have nothing to do with medical data
- Questions that are plausible but with no related table.

### [Answerable questions]
- Questions that are answerable with the given tables

### Here are some examples of answerable questions and the reason for why they are answerable:
"Can you list the yearly maximum vital high protein (full) input of patient 10019777?"
- The question is answerable because the d_items table links "vital high protein (full)" to inputevents, which records totalamount and starttime. The SQL query can be generated to correctly filters by patient ID, groups by year, and calculates the maximum input.
"What were the top five most common output events since 2100?"
- The question is answerable because "since 2100" includes all timestamps from the year 2100 onward, not just future data. If the dataset contains records from 2100 or later, valid results can be retrieved. 
"What is the number of times that patient 10018081 had ostomy (output) since 12/03/2100?"
- The question is answerable because "ostomy (output)" is a recorded event in the dataset, and the required patient identification and time filtering can be performed using available tables. The term "ostomy (output)" exists in the data schema, making it possible to count occurrences.

### And these examples of unanswerable questions:
"What is the meaning of life?"
"Who is the best basketball player of all time?"
"Is the sky blue?"

Finally, assign a score from 0 to 100 indicating how answerable the question is, where:
- 0 means completely unanswerable, which is vauge, or unrelated to the tables
- 50 means partially answerable or requires additional columns or data
- 100 means fully answerable with the given data

Present your response in the following format:

<analysis>
[Your detailed reasoning here]
</analysis>

<score>
[Your numerical score from 0 to 100]
</score>

Remember to provide thorough reasoning before giving the final score.

### User question:
{USER_QUESTION}
"""

answerable_classification_user_prompt_hard = """Your task is to classify the following user question as answerable or unanswerable based on the SQL tables and the assumptions:

### [Unanswerable questions]
- Vague questions
- Questions that have nothing to do with medical data
- Questions that are plausible but with no related table.

### [Answerable questions]
- Questions that are answerable with the given tables

### Here are some examples, where null means unanswerable:
{EXAMPLES}

Finally, classify the question as either Answerable or Unanswerable(null) based on the following criteria:

- Answerable: The question can be fully answered using the available tables and data
- null: The question is vague, unrelated to medical data, or cannot be answered with the given tables

Present your response in the following format:
<classification>
[Answerable/null]
</classification>

Remember to provide thorough reasoning before giving the final classification.

### User question:
{USER_QUESTION}
"""


sql_generation_system_prompt = """You are an AI assistant tasked with generating a SQL query to answer a given user question. Your job is to generate a valid SQL query that can be executed on the provided SQL tables.

### Here are the SQL tables:
{SQL_TABLES}

### Here is the assumptions:
{SQL_ASSUMPTIONS}
"""

sql_generation_user_prompt = """Your task is to generate a SQL query to answer the following user question:

NLQ: {USER_QUESTION} 
SQL:
""" 


sql_repair_system_prompt = """You are an AI assistant tasked with repairing a SQL query that was generated to answer a given user question. Your job is to identify and fix any errors in the SQL query that prevent it from executing correctly.

### Here are the SQL tables:
{SQL_TABLES}

### Here is the assumptions:
{SQL_ASSUMPTIONS}

"""

sql_repair_user_prompt = """Your task is to repair the following SQL query that was generated to answer the following user question:

NLQ: {USER_QUESTION} 

### Here is the original SQL query:
{SQL_QUERY}

### Here is the error message:
{ERROR_MESSAGE}

### SQL Query answer format:
```sql
{{NEW_SQL_QUERY_HERE}}
```
"""



