from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# https://python.langchain.com/docs/integrations/chat/llama2_chat#chat-with-llama-2-via-llamacpp-llm
# https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF#provided-files
llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.Q5_K_M.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

prompt = """
Question: Tell me about quantum mechanics. What is the spooky effect at a distance?
"""
llm(prompt)
