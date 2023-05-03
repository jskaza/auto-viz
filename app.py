import openai
import panel as pn
from sklearn import datasets
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

openai.api_key  = os.environ.get("OPENAI_KEY")
iris = datasets.load_iris()
type_map = {k: v for k, v in enumerate(iris['target_names'])}
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
df['target'] = df['target'].map(type_map)
numeric_vars = list(df.dtypes[df.dtypes == float].index)
other_vars = list(df.dtypes[df.dtypes != float].index)
cat_levels = {}
for var in other_vars:
    cat_levels[var] = list(df[var].unique())

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def collect_messages(_):
    prompt = inp.value_input
    inp.value = ''
    context.append({'role':'user', 'content':f"{prompt}"})
    response = get_completion_from_messages(context) 
    context.append({'role':'assistant', 'content':f"{response}"})
    if len(context) > 3:
        panels.append(
            pn.Row('User:', pn.pane.Markdown(prompt, width=600))
        )
        try:
            exec(json.loads(context[-1]['content'])['codegen'])
            panels.append(
                pn.Row('Assistant:', 
                    pn.pane.PNG(json.loads(context[-1]['content'])['filename'])
                    )
            )
            panels.append(
                pn.Row(pn.pane.HTML(f"<code>{json.loads(context[-1]['content'])['codegen']}</code>".replace('\n', '<br/>'),
                                         width=600, style={'background-color': '#F6F6F6'}))
            )
        except Exception as e:
            panels.append(
                pn.Row('Assistant:', pn.pane.Markdown(f'Unable to generate plot: {str(e)}', width=600, style={'background-color': '#F6F6F6'}))
            )
    
        return pn.Column(*panels)

f = open('prompt.txt', 'r')
prompt = f.read()
prompt = prompt.replace('NUMERIC_VARS', str(numeric_vars))
prompt = prompt.replace('CAT_LEVELS', str(cat_levels))
prompt = prompt.replace('NUMERIC_VAR_1', numeric_vars[0])
prompt = prompt.replace('NUMERIC_VAR_2', numeric_vars[1])
prompt = prompt.replace('CAT_VAR_1', list(cat_levels.keys())[0])
# print(prompt)

pn.extension()

panels = [] # collect display 

context = [ {'role':'system', 'content':prompt} ]

inp = pn.widgets.TextInput(value="Hi", placeholder='Enter description of a plot here…')
button_conversation = pn.widgets.Button(name='Chat!')

interactive_conversation = pn.bind(collect_messages, button_conversation)

dashboard = pn.Column(
    pn.Row('Visually explore the Iris Dataset using Natural Language prompts.'),
    pn.Row(pn.pane.HTML(df.describe(include='all').to_html())),
    inp,
    pn.Row(button_conversation),
    pn.panel(interactive_conversation, loading_indicator=True, height=300),
)

dashboard.show()

