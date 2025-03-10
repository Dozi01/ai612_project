from openai import OpenAI, AsyncOpenAI
import time
import asyncio
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

MODEL_DICT = {
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
}


class OpenAIModel:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float,
        async_mode: bool = True,
        **kwargs,
    ):

        try:
            self.async_model = AsyncOpenAI(api_key=api_key)
            self.model = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Init openai client error: \n{e}")
            raise RuntimeError("Failed to initialize OpenAI client") from e

        self.model_name = MODEL_DICT[model_name]
        self.temperature = temperature
        self.async_mode = async_mode

        if model_name in MODEL_DICT.keys():
            self.batch_forward_func = self.batch_forward_chatcompletion
            self.generate = self.gpt_chat_completion
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def batch_forward_chatcompletion(self, batch_prompts):
        """
        Input a batch of prompts to openai chat API and retrieve the answers.
        batch mode is activated when the batch_mode is true and batch has more than 10 prompts.
        """
        if self.async_mode:
            responses = self.async_generate_responses(batch_prompts=batch_prompts)
        else:
            responses = []
            for prompt in tqdm(batch_prompts):
                response = self.gpt_chat_completion(prompt=prompt)
                responses.append(response)
        return responses

    def gpt_chat_completion(self, prompt):
        backoff_time = 1
        while True:
            try:
                return (
                    self.model.chat.completions.create(
                        messages=prompt,
                        model=self.model_name,
                        temperature=self.temperature,
                    )
                    .choices[0]
                    .message.content.strip()
                )
            except Exception as e:
                print(e, f" Sleeping {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5

    # functions for async mode.
    async def _generate_single_prompt(self, prompt):
        """
        Generate response for a single prompt asynchronously.
        """
        backoff_time = 1
        while backoff_time < 30:
            try:
                response = await self.async_model.chat.completions.create(
                    messages=prompt,
                    model=self.model_name,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(e, f" Sleeping {backoff_time} seconds...")
                await asyncio.sleep(backoff_time)
                backoff_time *= 2

    async def async_generation(self, batch_prompts, chunk_size=300):
        responses = []
        for i in range(0, len(batch_prompts), chunk_size):
            batch = batch_prompts[i : i + chunk_size]
            tasks = [self._generate_single_prompt(prompt) for prompt in batch]
            batch_responses = await tqdm_asyncio.gather(*tasks)
            responses.extend(batch_responses)
        return responses

    def async_generate_responses(self, batch_prompts):
        return asyncio.run(self.async_generation(batch_prompts))
