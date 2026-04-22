"""
Lightweight RAG (Retrieval-Augmented Generation) Layer
=======================================================
Retrieves relevant papers and formats them for LLM consumption.
Supports multiple LLM backends (OpenAI, Gemini, local/Ollama).
"""

from typing import Optional
import os
import pandas as pd

from src.recommender import ContentRecommender


def retrieve_context(
    query: str,
    recommender: ContentRecommender,
    top_k: int = 5,
) -> tuple[pd.DataFrame, str]:
    """
    Retrieve relevant papers and format as context for LLM.

    Parameters
    ----------
    query : str
        Natural language query
    recommender : ContentRecommender
        Initialized recommender
    top_k : int
        Number of papers to retrieve

    Returns
    -------
    tuple[pd.DataFrame, str]
        (results_df, formatted_context)
    """
    results = recommender.recommend_by_query(query, top_k=top_k)

    # Format context for LLM prompt
    context_parts = []
    for i, row in results.iterrows():
        authors = ", ".join(row["authors"][:5]) if isinstance(row["authors"], list) else "Unknown"
        context_parts.append(
            f"[{i+1}] Title: {row['title']}\n"
            f"    Authors: {authors}\n"
            f"    Year: {row.get('year', 'N/A')} | Venue: {row.get('venue', 'N/A')}\n"
            f"    Relevance Score: {row.get('similarity_score', 0):.4f}"
        )

    formatted_context = "\n\n".join(context_parts)
    return results, formatted_context


def build_prompt(
    query: str,
    context: str,
    task: str = "summarize",
) -> str:
    """
    Build an LLM prompt with retrieved context.

    Parameters
    ----------
    query : str
        User's question
    context : str
        Formatted paper context
    task : str
        'summarize', 'trends', 'compare', or 'explain'

    Returns
    -------
    str
        Complete LLM prompt
    """
    task_instructions = {
        "summarize": (
            "Summarize the key findings and contributions of these papers. "
            "Highlight common themes and methodologies."
        ),
        "trends": (
            "Analyze trends across these papers. What research directions are emerging? "
            "How has the field evolved based on publication years?"
        ),
        "compare": (
            "Compare and contrast the approaches taken in these papers. "
            "What are the similarities and differences?"
        ),
        "explain": (
            "Explain how these papers relate to the query. "
            "What insights can be drawn from this body of work?"
        ),
    }

    instruction = task_instructions.get(task, task_instructions["summarize"])

    prompt = f"""You are a research assistant analyzing computer science publications from DBLP.

Based on the following retrieved papers relevant to the query "{query}":

{context}

{instruction}

Provide a clear, well-structured response with concrete references to the papers listed above.
Use the paper numbers [1], [2], etc. when referencing specific papers."""

    return prompt


def generate_response_openai(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    temperature: float = 0.3,
) -> str:
    """Generate response using OpenAI API."""
    try:
        import openai

        if api_key:
            openai.api_key = api_key
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            openai.api_key = api_key

        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except ImportError:
        return "OpenAI package not installed. Run: uv add openai"
    except Exception as e:
        return f"OpenAI API error: {e}"


def generate_response_gemini(
    prompt: str,
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> str:
    """Generate response using Google Gemini API."""
    try:
        from google import genai

        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            api_key = os.getenv("GOOGLE_API_KEY")
            client = genai.Client(api_key=api_key)    

        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text
    except ImportError:
        return "google-genai package not installed. Run: uv add google-genai"
    except Exception as e:
        return f"Gemini API error: {e}"


def rag_query(
    query: str,
    recommender: ContentRecommender,
    llm_backend: str = "gemini",
    task: str = "summarize",
    top_k: int = 5,
    **kwargs,
) -> dict:
    """
    Full RAG pipeline: retrieve + generate.

    Parameters
    ----------
    query : str
        Natural language query
    recommender : ContentRecommender
        Initialized recommender
    llm_backend : str
        'openai', 'gemini', or 'ollama'
    task : str
        Task type for prompt construction
    top_k : int
        Number of papers to retrieve

    Returns
    -------
    dict
        {
            'query': original query,
            'retrieved_papers': DataFrame of retrieved papers,
            'context': formatted context string,
            'prompt': full prompt sent to LLM,
            'response': LLM-generated response
        }
    """
    # Retrieve
    results, context = retrieve_context(query, recommender, top_k=top_k)

    # Build prompt
    prompt = build_prompt(query, context, task=task)

    # Generate
    generators = {
        "openai": generate_response_openai,
        "gemini": generate_response_gemini,
    }

    generator = generators.get(llm_backend)
    if generator is None:
        response = f"Unknown backend: {llm_backend}. Use 'openai', 'gemini'."
    else:
        response = generator(prompt, **kwargs)

    return {
        "query": query,
        "retrieved_papers": results,
        "context": context,
        "prompt": prompt,
        "response": response,
    }
