import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks import get_openai_callback


def init_data(dataset):
    """
    Takes in .csv dataset as input and 
    returns pandas DataFrame and a 10-row sample of the data
    """
    if dataset:
        df = pd.read_csv(dataset)
    else: 
        df = pd.read_csv('./data/car_sales.csv')

    data_sample = df.head(10).to_csv()

    return df, data_sample


def init_model(OPENAI_API_KEY, temperature):
    """
    Takes OPENAI_API_KEY and temperature as input and 
    returns intialized model
    """
    try:
        chat_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo', temperature=temperature)
        return chat_llm
    except ValueError:
        return None
    

def generate_prompt(num_charts, data_sample):
    """
    Takes in number of plots to generate and a sample of data as a csv string 
    and returns ChatModel prompt for data visualization.
    """

    system_template = """/
        The following is a conversation between a Human and an AI assistant expert on data visualization with perfect Python 3 syntax. The human will provide a sample dataset for the AI to use as the source. The real dataset that the human will use with the response of the AI is going to have several more rows. The AI assistant will only reply in the following JSON format: 

        {{ 
        "charts": [{{'title': string, 'chartType': string, 'parameters': {{...}}}}, ... ]
        }}

        Instructions:

        1. chartType must only contain methods of plotly.express from the Python library Plotly.
        2. The format for charType string: plotly.express.chartType.
        3. For each chartType, parameters must contain the value to be used for all parameters of that plotly.express method.
        4. There should 4 parameters for each chart.
        5. Do not include "data_frame" in the parameters.
        6. There should be {num_charts} charts in total.
        """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = """
        Human: 
        This is the dataset:

        {data}
        """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    prompt = chat_prompt.format_prompt(num_charts=str(num_charts), data=data_sample).to_messages()
    
    return prompt


def generate_result(chat_llm, prompt):
    """
    Takes ChatModel and prompt as input and returns formatted output
    """
    with get_openai_callback() as cb:
            result = chat_llm(prompt)
    total_token = cb.total_tokens
    total_cost = cb.total_cost

    # format result to output
    output = json.loads(result.content)

    return (output, total_token, total_cost)
    