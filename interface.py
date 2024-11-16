import gradio as gr
from chat import generate_response

# Define the Gradio interface function
def chat_interface(user_input):
    return generate_response(user_input)

# Create the Gradio interface
iface = gr.Interface(
    fn=chat_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your command or question here..."),
    outputs=gr.Textbox(label="TinyLlama Response"),
    title="TinyLlama Command Assistant",
    description="Interact with the TinyLlama model for generating responses.",
)

if __name__ == "__main__":
    iface.launch()
