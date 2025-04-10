answerable_classification_system_prompt = """You are an AI assistant tasked with determining whether a given question can be answered using the provided SQL tables. Your job is to classify the question as answerable or unanswerable and provide a score indicating how answerable it is.

### Examine the SQL tables:
{SQL_TABLES}

### Here are the rules that you must follow:
{SQL_ASSUMPTIONS}

"""

answerable_classification_user_prompt = """Your task is to classify the following user question as answerable or unanswerable based on the SQL tables and the rules:

### [Unanswerable questions]
- Vague questions
- Questions that have nothing to do with medical data
- Questions that have no relation to the tables
- Questions that are not answerable with the given data
- Questions that are plausible but with no related table.

### [Answerable questions]
- Questions that are answerable with the given tables

### Here are some examples of answerable questions:
[Example 1]
- User question: "What was the prescription drug that patient 10004235 was prescribed for within the same hospital visit after receiving laparoscopic robotic assisted procedure since 03/2100?"
- Response:
    <analysis>We have patient, procedures_icd, prescriptions tables, which can fully support the question.</analysis>
    <result>YES</result>
[Example 2]
- User question: "What is patient 10021487's monthly average bilirubin, direct levels since 05/2100?"
    <analysis>We have patient, lab, and other tables, which can fully support the question.</analysis>
    <result>YES</result>
[Example 3]
- User question: "What is the average number of days between a patient's first and second hospital visits?"
    <analysis>We have patient, admissions, and other tables, which can fully support the question.</analysis>
    <result>YES</result>

### And these examples of unanswerable questions:
[Example 1]
- User question: "What is the meaning of life?"
    <analysis>Meaning of life is vague, and has nothing to do with medical data.</analysis>
    <result>NO</result>
[Example 2]
- User question: "Who is the best basketball player of all time?"
    <analysis>Basketball has nothing to do with medical data.</analysis>
    <result>NO</result>
[Example 3]
- User question: "What are the record companies with patients older than 45?"
    <analysis>It seems plausible, but there are no table that is related to record companies.</analysis>
    <result>NO</result>

### Response format
Decide whether the question is answerable or not.
- Return 'YES' for answerable
- Return 'NO' for unanswerable

Present your response in the following format:

<analysis>
[Your detailed reasoning here]
</analysis>

<result>
[YES or NO]
</result>

Remember to provide thorough reasoning before giving the final answer.

### User question:
{USER_QUESTION}
"""


sql_generation_system_prompt = """You are an AI assistant tasked with generating a SQL query to answer a given user question. Your job is to generate a valid SQL query that can be executed on the provided SQL tables.

### Here are the SQL tables:
{SQL_TABLES}

### Here are the rules that you must follow:
{SQL_ASSUMPTIONS}
"""

sql_generation_user_prompt = """Your task is to generate a SQL query to answer the following user question:

Remember to follow the rules.

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

Remember to follow the rules.

### SQL Query answer format:
```sql
{{NEW_SQL_QUERY_HERE}}
```
"""



