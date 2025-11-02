from dotenv import load_dotenv
import os, time, gradio as gr
from openai import OpenAI
import mlflow

load_dotenv(override=True)

# Configuración de modelos en la nube

from collections import namedtuple
from functools import reduce

MAX_TOKENS = 500

Provider = namedtuple('Provider', ['name', 'base_url', 'api_key', 'models'])
Task = namedtuple('Task', ['name', 'prompt'])

providers = [
    Provider(
        'open_router', 'https://openrouter.ai/api/v1', 'OPEN_ROUTER_API_KEY',
        ["gpt-3.5-turbo", "deepseek/deepseek-r1:free", "gpt-4o-mini"]
    ),
    Provider(
        'gemini', 'https://generativelanguage.googleapis.com/v1beta/openai/', 'GEMINI_API_KEY',
        ["gemini-2.0-flash"]
    ),
    Provider(
        'groq', 'https://api.groq.com/openai/v1', 'GROQ_API_KEY',
        ["llama-3.3-70b-versatile"]
    ),
]

translation_task = Task('translation', 'Translate the following text to spanish, provide only the translation: processing_text')
summarization_task = Task('summarization', 'Summarize the following text to number_of_words words, provide only the summarization, and let the language as is: processing_text')

tasks = [translation_task, summarization_task]

model_names = list(reduce(lambda x, y: [*x, *y], map(lambda x: x.models, providers), []))
task_names = list(map(lambda x: x.name, tasks))

import time

def get_prompt(task_name, text, number_of_words=None):
    task = next(filter(lambda x: x.name == task_name, tasks))
    prompt: str = task.prompt
    prompt = prompt.replace('number_of_words', str(number_of_words))
    prompt = prompt.replace('processing_text', text)
    return prompt

def get_provider_api(model):
    provider = next(filter(lambda x: model in x.models, providers))
    api_key = os.getenv(provider.api_key)
    api = OpenAI(api_key=api_key, base_url=provider.base_url)
    return api


def process_text(model, task, text, number_of_words=None):
    if not text.strip():
        yield "Provide a text please...", None, None
    
    prompt = get_prompt(task, text, number_of_words)
    prompt_view = gr.update(value=f'__Prompt:__ {prompt}', visible=True)

    yield None, None, prompt_view
    start_time = time.time()
    
    try:
        provider_api = get_provider_api(model)
        response = provider_api.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=MAX_TOKENS
        )
    except Exception as e:
        yield f"⚠️ Error connecting to {provider_api.base_url}: {str(e)}", None, None
        return
        

    output = response.choices[0].message.content
    end_time = time.time()
    elapsed = round(end_time - start_time, 2)

    # Registar run en MLflow
    mlflow.start_run()
    mlflow.set_tag("model_type", task)
    mlflow.log_param("model", model)
    mlflow.log_param("task", task)
    mlflow.log_param("text", text)
    mlflow.log_param("Number of words", number_of_words)
    mlflow.log_param("text_length", len(text) if text else 0)
    mlflow.log_metric("response_time_seconds", elapsed)
    mlflow.log_metric("output_length", len(output) if output else 0)
    mlflow.log_outputs({"output": output})
    mlflow.end_run()

    yield output, f'{elapsed}s', prompt_view


# Interfaz con gradio

css = """
.center-text {
    text-align: center;
}
.contain {
    justify-content: center;
}
.truncated {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
#container {
    max-width: 900px;
    margin: 0 auto;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id='container'):
        gr.Markdown('# ChatMC', elem_classes=['center-text'])
        gr.Markdown('*What\'s up?*', elem_classes=['center-text'])
        
        prompt_content = gr.Markdown(visible=False, elem_classes=['center-text', "truncated"])
        
        with gr.Row(equal_height=True):
            with gr.Column():
                input = gr.TextArea(placeholder='Input text...', label='Input')

            with gr.Column():
                output = gr.TextArea(
                    placeholder='Output will appear here...', 
                    label='Output', 
                    show_copy_button=True
                )

        with gr.Row(equal_height=True):
            with gr.Column():
                process_button = gr.Button("Process")
            with gr.Row(equal_height=True):
                elapsed_time = gr.Label(label='Processing time')
                use_as_input = gr.Button('Use as Input')
        
        with gr.Row():
            model = gr.Dropdown(model_names, label="Model")
            task = gr.Dropdown(task_names, label='Task')
            number_of_words = gr.Number(None, label='Max number of words', visible=False)
    
    # It's needed to hide 'Max number of words' options when not using summarization
    def on_change_task(task):
        return gr.update(visible=(task == summarization_task.name), value=None)
                         
    task.change(on_change_task, inputs=[task], outputs=[number_of_words])
    use_as_input.click(lambda x: x, inputs=[output], outputs=[input])
    process_button.click(process_text, inputs=[model, task, input, number_of_words], outputs=[output, elapsed_time, prompt_content])


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://mlflow:5000")  
    mlflow.set_experiment("llm-chat")
    demo.launch(server_name="0.0.0.0", server_port=7860)
