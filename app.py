import streamlit as st
import utils
import plotly.express


st.title("📊 AutoDataViz")


# API key
OPENAI_API_KEY = st.text_input('OpenAI API Key')


# hyperparameters
temp = st.number_input("Temperature", min_value=0.0, max_value=0.9, value=0.0, step=0.1)
num_charts = st.slider("Number of Plots", min_value=1, max_value=10, value=4)


# data
dataset = st.file_uploader("Upload your dataset", accept_multiple_files=False, type='csv')
df, data_sample = utils.init_data(dataset)
st.dataframe(df)


# model
chat_llm = utils.init_model(OPENAI_API_KEY, temp)


if st.button("Generate"):
    if not chat_llm:
        st.error('Error: Please add a valid OpenAI API Key.', icon="🚨")
    else:
        st.markdown("## Visualizations")

        # prompt
        prompt = utils.generate_prompt(num_charts, data_sample)

        # generate result
        (output, total_token, total_cost) = utils.generate_result(chat_llm, prompt)

        # tokens and cost
        st.markdown(f"#### Tokens: {total_token}")
        st.markdown(f"#### Cost (USD): {total_cost}")
        
        # generate plots
        for i in range(num_charts):
            try:
                code = f"""
import plotly

params = {output['charts'][i]['parameters']}
params['data_frame'] = df
params['title'] = "{output['charts'][i]['title']}"

fig = {output['charts'][i]['chartType']}(**params)
    """

                exec(code)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.markdown(f"__{output['charts'][i]['title']}: Visualization Failed__")
                st.text(output['charts'][i])




