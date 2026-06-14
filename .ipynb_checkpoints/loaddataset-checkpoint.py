from datasets import load_dataset
import ollama
import requests
import json
import time
import jsonlines
import os
from dotenv import load_dotenv
import re

img_url = "https://image.tmdb.org/t/p/w600_and_h900_face"

def load_tmdb_movies(max_films=None):
    dataset = load_dataset("ada-datadruids/full_tmdb_movies_dataset", split="train")
    img_url = "https://image.tmdb.org/t/p/w600_and_h900_face"
    movies = []
    for i, item in enumerate(dataset):
        if max_films and i >= max_films:
            break

        title = item.get("title")
        date_str = item.get("release_date")
        if not title or not date_str:
            continue

        year = int(date_str[:4]) if len(date_str) >= 4 else None
        if year is None:
            continue

        genres = item.get("genres", [])
        genre_str = ""
        if genres:
            if isinstance(genres[0], dict):
                genre_names = [g.get("name", "") for g in genres if g.get("name")]
            else:
                genre_names = [str(g) for g in genres]
            genre_str = "".join(genre_names)

        overview = item.get("overview", "")
        if not overview:
            continue

        tagline = item.get("tagline") or ""
        keyword = item.get("keyword") or "" 
        vote = item.get("vote_average")
        vote_count = item.get("vote_count", 0)
        
        if not vote or vote < 6.5 or vote_count < 1000:
            continue

        poster_path = item.get("poster_path")
        poster_url = img_url + poster_path if poster_path else None

        synopsis = f"Genres: {genre_str}. Overview: {overview}. Tagline: {tagline}. Keywords: {keyword}"

        movies.append({
            "title": title,
            "year": year,
            "release_date": date_str,
            "synopsis": synopsis,
            "poster_url": poster_url,
            "vote": vote,
        })
    return movies

movies = load_tmdb_movies()

def build_prompt(batch):
    movies_block = ""
    for i, movie in enumerate(batch, 1):
        movies_block += f"{i}. Title: {movie['title']} ({movie['year']})\n"
        movies_block += f"   Synopsis: {movie['synopsis']}\n\n"

    prompt = f"""<s>[INST] You are a dataset generator for a movie recommendation AI.
For each movie below, generate 7 realistic user requests in different styles.
These requests are what a person would type into a search bar to find a movie they WANT TO WATCH.
The CORRECT recommendation for all requests is the exact movie provided.

Style definitions (follow them strictly, produce at least one full sentence, NOT just a single word):
- plot_request     : describe the desired plot in 1-2 sentences, without spoiling the ending.
- vibe_request     : ask for a movie with a certain atmosphere, feeling or visual style. Make it descriptive.
- genre_request    : state the genre(s) you're in the mood for, as a full sentence (e.g., "I'm looking for a sci-fi thriller with mind-bending twists").
- half_baked_idea  : very vague idea of what you want, with hesitation, incomplete thoughts (like "I don't know, something with... maybe time travel?"). Use casual language.
- slang_request    : very casual, internet slang, abbreviations, intentional spelling/grammar mistakes. Sound like a teenager texting. No emojis or special characters.
- emotional_need   : focus on the emotion you want to feel (e.g., "a movie that will make me cry happy tears" or "something that fills me with wonder and hope"). Full sentence.
- title_request    : write the exact movie title followed by 3 to 5 keywords that directly describe the movie (genre, mood, main theme). No full sentence needed.

CRITICAL RULES:
- The "title" field in your output JSON must be the EXACT ORIGINAL title from the input, unchanged.
- NEVER mention the title of the given movie or any other specific movie title inside the descriptions, EXCEPT in the title_request style where the exact title must appear.
- NEVER use phrases like "like X but...".
- Make each request unique; vary vocabulary and sentence structures.
- Do NOT output single-word answers. Every request except title_request must be at least 6 words.
- Do NOT copy the synopsis verbatim, rephrase completely.
- Do NOT use emojis, emoticons, or special characters.

Movies:
{movies_block}
Return a JSON array with exactly {len(batch)} objects, one per movie.
Each object must have: movie_id (int), title (str), year (int), descriptions (array of 7 objects with style and text fields).
No extra text, no markdown, just the raw JSON array. [/INST]"""
    return prompt


def call_ollama(prompt, retries=3):
    for attempt in range(1, retries + 1):
        if attempt > 1:
            wait = 3
            print(f" Retry {attempt}/{retries} (waiting {wait}s...)")
            time.sleep(wait)
        t0 = time.time()

        try:
            response = ollama.generate(
                model="mistral",
                prompt=prompt,
            )
            elapsed = time.time() - t0
            raw = response['response'].strip()

            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON array found in response")

            batch_results = json.loads(raw[start:end])
            print(f"Done in {elapsed:.1f}s — {len(batch_results) * 7} pairs generated")
            return batch_results

        except Exception as e:
            elapsed = time.time() - t0
            print(f" Failed after {elapsed:.1f}s: {e}")

    return []


def save_to_jsonl(batch_results, batch_movies, path="dataset.jsonl"):
    movie_map = {(m["title"], m["year"]): m for m in batch_movies}
    with jsonlines.open(path, mode="a") as writer:
        for movie in batch_results:
            title = movie["title"]
            year = movie["year"]
            meta = movie_map.get((title, year), {})
            for desc in movie.get("descriptions", []):
                if (
                    not isinstance(desc, dict)
                    or "text" not in desc
                    or "style" not in desc
                    or not isinstance(desc["text"], str)
                    or len(desc["text"].split()) < 4
                ):
                    continue
                writer.write({
                    "input": desc["text"],
                    "output": f"{title} ({year})",
                    "style": desc["style"],
                    "title": title,
                    "year": year,
                    "release_date": meta.get("release_date"),
                    "poster_url": meta.get("poster_url"),
                    "vote": meta.get("vote"),
                })

def generate_dataset(movies, batch_size=6, output_path="dataset.jsonl"):
    done = set()
    try:
        with jsonlines.open(output_path) as reader:
            for row in reader:
                done.add(row["output"])
    except FileNotFoundError:
        pass

    if done:
        print(f"Resuming — {len(done)} movies already done, skipping them.\n")

    total = (len(movies) + batch_size - 1) // batch_size

    for i in range(0, len(movies), batch_size):
        raw_batch = movies[i:i + batch_size]
        batch = [m for m in raw_batch if f"{m['title']} ({m['year']})" not in done]
        batch_num = i // batch_size + 1

        if not batch:
            print(f"[{batch_num}/{total}] Already done, skipping")
            continue

        print(f"\n[{batch_num}/{total}] {[m['title'] for m in batch]}")

        results = call_ollama(build_prompt(batch))
        if results:
            save_to_jsonl(results, batch, path=output_path)
        else:
            print("Batch skipped after all retries")

        if i + batch_size < len(movies):
            time.sleep(0.2)

    print("Completed")


print(f"{len(movies)} films chargés")
generate_dataset(movies, batch_size=6)
