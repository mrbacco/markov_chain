#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Markov‑chain text generator that *learns* its corpus from an LLM.

Features
--------
* Works with any LLM that can be called through a Python function
  (OpenAI, Anthropic, Azure, Hugging‑Face, local Transformers, …).
* Supports arbitrary order‑k Markov models (default = 2‑grams).
* Generates grammatically‑reasonable sentences because the source
  data comes from a powerful LLM.
* No external data files required – the corpus is created on‑the‑fly.

Author  :  mrbacco04@gmail.com
Created :  20266 - Q2
"""

# ------------------------------------------------------------
# 1️⃣  Imports
# ------------------------------------------------------------
import os
import re
import random
import itertools
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Iterable

# ------------------------------------------------------------
# 2️⃣  LLM interface
# ------------------------------------------------------------
# -----------------------------------------------------------------
#   👉 Choose ONE of the three ready‑made functions below.
# -----------------------------------------------------------------
#   • `openai_generate` – uses the OpenAI Chat API (gpt‑3.5‑turbo,
#                         gpt‑4, Claude, etc. – just change the model name)
#   • `hf_generate`    – uses a Hugging‑Face text‑generation pipeline
#                         (e.g. "gpt2", "EleutherAI/gpt‑neo-2.7B")
#   • `dummy_generate` – returns a static sample (useful for offline testing)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# 2️⃣a  OpenAI (or any compatible chat‑completion API)
# -----------------------------------------------------------------
def openai_generate(prompt: str,
                    model: str = "gpt-3.5-turbo",
                    max_tokens: int = 1024,
                    temperature: float = 0.9,
                    n: int = 1) -> List[str]:
    """
    Call the OpenAI chat/completion endpoint and return the generated texts.

    The function expects an environment variable `OPENAI_API_KEY` with a
    valid key.  If you want to use Anthropic or another provider that
    follows the same payload style, just replace `openai.ChatCompletion`
    with the appropriate client.
    """
    import openai  # lazy import – only needed if you pick this backend

    # Ensure the key is available (OpenAI will also raise its own error)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable.")
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant that writes short, varied English sentences."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        stop=None  # let the model decide; you can add "\n\n" etc. here
    )

    # The API returns a list of choices, each with a Message object.
    return [choice["message"]["content"] for choice in response["choices"]]


# -----------------------------------------------------------------
# 2️⃣b  Hugging‑Face Transformers (local or remote)
# -----------------------------------------------------------------
def hf_generate(prompt: str,
                model_name: str = "gpt2",
                max_new_tokens: int = 512,
                temperature: float = 0.9,
                seed: int = None,
                device: int = -1) -> List[str]:
    """
    Generate text using the Hugging‑Face `pipeline` API.
    Set `device=0` (or another GPU id) to run on a GPU,
    otherwise `-1` forces CPU execution.
    """
    from transformers import pipeline, set_seed

    generator = pipeline(
        "text-generation",
        model=model_name,
        device=device,
        # Add any tokenizer / config tweaks here, e.g. `torch_dtype=torch.float16`
    )
    if seed is not None:
        set_seed(seed)

    # The pipeline returns a list of dicts with a `generated_text` key.
    out = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    return [x["generated_text"] for x in out]


# -----------------------------------------------------------------
# 2️⃣c  Dummy generator (offline debugging)
# -----------------------------------------------------------------
def dummy_generate(prompt: str, **_: dict) -> List[str]:
    """
    Very small static fallback – useful if you have no internet.
    """
    sample = (
        "The cat chased the red ball. "
        "A quiet river flows through the forest. "
        "She reads a book while sipping tea. "
        "Lightning illuminated the night sky. "
        "Dreams whisper softly in the dark."
    )
    # Return the same static sample wrapped into a list.
    return [sample]


# -----------------------------------------------------------------
# 2️⃣d  Choose which backend you want to use.
# -----------------------------------------------------------------
# Uncomment ONE of the three lines below, then use the name `generate_text()`
# throughout the rest of the script.

# generate_text = openai_generate   # <-- OpenAI / Anthropic / Azure Chat API
# generate_text = hf_generate      # <-- Hugging Face (local) models
generate_text = dummy_generate    # <-- No‑internet placeholder (keep for testing)


# ------------------------------------------------------------
# 3️⃣  Helper: ask the LLM for a *corpus* of raw sentences
# ------------------------------------------------------------
def request_corpus(num_sentences: int = 500,
                   max_tokens_per_call: int = 1024,
                   temperature: float = 0.9) -> str:
    """
    Ask the LLM to produce `num_sentences` short, human‑readable English
    sentences, one per line.  The function may need to make several calls
    if the requested token budget exceeds what a single request can return.
    """
    # Prompt engineering – keep it brief but explicit.
    prompt = (
        f"Please write {num_sentences} short, varied English sentences, "
        "one sentence per line.  Do not include any numbering or bullet points."
    )
    # Most LLMs will return the whole block in one go as long as max_tokens is high enough.
    # We therefore ask for a *large* token budget.  If the provider caps us,
    # a loop can be added to fetch more chunks – omitted for brevity.
    raw_responses = generate_text(
        prompt,
        max_tokens=max_tokens_per_call,
        temperature=temperature,
        n=1,
    )
    # The API returns a list – we just need the first (and only) element.
    corpus_text = raw_responses[0]
    return corpus_text


# ------------------------------------------------------------
# 4️⃣  Tokenisation utilities
# ------------------------------------------------------------
_TOKEN_RE = re.compile(r"\b\w+(?:'\w+)?\b|[.!?]")  # words + punctuation tokens


def simple_tokenise(text: str) -> List[str]:
    """
    Very lightweight tokeniser:
        - lower‑case the text
        - split on word boundaries and keep sentence‑ending punctuation
    It purposefully discards commas, quotes, dashes etc. – you can
    extend the regex if you need them.
    """
    # Normalise whitespace first
    text = re.sub(r"\s+", " ", text.strip())
    tokens = _TOKEN_RE.findall(text.lower())
    return tokens


def split_into_sentences(text: str) -> List[List[str]]:
    """
    Return a list of token lists, one per sentence.
    The function keeps the ending punctuation token (.,!?) attached to its
    sentence because it is useful for start‑state detection.
    """
    # This implementation uses simple punctuation heuristics.
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [simple_tokenise(s) for s in raw_sentences if s]


# ------------------------------------------------------------
# 5️⃣  Markov‑chain model (order‑k)
# ------------------------------------------------------------
class MarkovChain:
    """
    Variable‑order (k‑gram) Markov chain.
    - `order` = length of the state (e.g. order=2 ⇒ bi‑gram model).
    - `transitions` maps `state(tuple)` → Counter(next_token).
    - `starts` holds a list of observed start states (first `order` tokens of each sentence).
    """

    def __init__(self, order: int = 2):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order: int = order
        self.transitions: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.starts: List[Tuple[str, ...]] = []  # observed sentence‑starts

    # -----------------------------------------------------------------
    # 5️⃣a  Training / learning
    # -----------------------------------------------------------------
    def train(self, tokenised_sentences: Iterable[List[str]]) -> None:
        """
        Populate `self.transitions` and `self.starts` from an iterable of token lists.
        """
        for sentence in tokenised_sentences:
            # Guard against empty sentences
            if len(sentence) < self.order:
                continue

            # Record the first `order` tokens as a possible start state:
            start_state = tuple(sentence[: self.order])
            self.starts.append(start_state)

            # Slide a window of length `order` across the sentence
            for i in range(len(sentence) - self.order):
                state = tuple(sentence[i : i + self.order])
                next_tok = sentence[i + self.order]
                self.transitions[state][next_tok] += 1

            # Optionally add a terminal state after the last token
            # (makes the generator stop naturally when it reaches end‑of‑sentence).
            terminal_state = tuple(sentence[-self.order :])
            self.transitions[terminal_state]["<EOS>"] += 1

    # -----------------------------------------------------------------
    # 5️⃣b  Helper – pick a next token from a state using learned probs
    # -----------------------------------------------------------------
    def _sample_next(self, state: Tuple[str, ...]) -> str:
        """
        Return a token sampled proportionally to its observed frequency.
        If the state is unknown (should not happen often), fallback to a random start.
        """
        if state not in self.transitions:
            # Unknown state – restart from a random start
            return random.choice(list(self.transitions.keys()))[0]

        next_counter = self.transitions[state]
        tokens, freqs = zip(*next_counter.items())
        total = sum(freqs)
        probs = [f / total for f in freqs]
        return random.choices(tokens, weights=probs, k=1)[0]

    # -----------------------------------------------------------------
    # 5️⃣c  Generation
    # -----------------------------------------------------------------
    def generate(self,
                 max_len: int = 30,
                 start_state: Tuple[str, ...] = None,
                 deterministic: bool = False) -> str:
        """
        Generate a single sentence.
        - `max_len` limits the number of tokens (including punctuation).
        - If `start_state` is None a random observed start is used.
        - If `deterministic=True` take the most frequent continuation at each step
          (useful for debugging).
        Returns a single human‑readable sentence (capitalised & punctuation cleaned).
        """
        # Choose a start state
        if start_state is None:
            state = random.choice(self.starts)
        else:
            state = tuple(start_state)

        # Initialise output tokens with the start state
        generated = list(state)

        for _ in range(max_len):
            if state not in self.transitions:
                break  # dead‑end – unlikely but safety first

            # Choose next token
            if deterministic:
                # Most common continuation
                next_tok = self.transitions[state].most_common(1)[0][0]
            else:
                next_tok = self._sample_next(state)

            if next_tok == "<EOS>":
                break  # reached end of sentence boundary

            generated.append(next_tok)

            # Shift the state window forward
            state = tuple(generated[-self.order :])

            # Stop early if we hit a sentence‑final punctuation token
            if next_tok in {".", "!", "?"}:
                break

        # -----------------------------------------------------------------
        # 5️⃣d  Post‑processing – prettify
        # -----------------------------------------------------------------
        # Join tokens with a space, then fix spacing before punctuation
        raw = " ".join(generated)
        # Remove spaces before .,!? and collapse multiple spaces
        cleaned = re.sub(r"\s+([.!?])", r"\1", raw)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Capitalise the first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned


# ------------------------------------------------------------
# 6️⃣  Main workflow
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Markov-chain text generator (LLM-based corpus)")
    print("=" * 60)

    # Step 1:  Get the corpus from the LLM
    print("\n[1/3] Requesting corpus from LLM...")
    try:
        corpus = request_corpus(num_sentences=50, max_tokens_per_call=1024)
        print(f"      {len(corpus.split())} words received.")
    except Exception as e:
        print(f"      Error: {e}")
        corpus = " ".join([
            "The cat sat on the mat.",
            "A swift brown fox jumped over the lazy dog.",
            "She walks her dog every morning.",
            "The sun sets in the west.",
            "Books are a source of knowledge."
        ])
        print(f"      Using fallback corpus.")

    # Step 2:  Tokenise and train the Markov chain
    print("\n[2/3] Training Markov chain (order=2)...")
    sentences = split_into_sentences(corpus)
    chain = MarkovChain(order=2)
    chain.train(sentences)
    print(f"      Learned {len(chain.transitions)} states.")

    # Step 3:  Generate new sentences
    print("\n[3/3] Generating sentences...")
    print("-" * 60)
    for i in range(5):
        sentence = chain.generate(max_len=25, deterministic=False)
        print(f"  {i + 1}. {sentence}")
    print("-" * 60)
    print("\nDone!")
