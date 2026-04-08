# movierec-two-towers

A retrieval-stage movie recommendation system built with a Two-Tower model, trained on the MovieLens 100K dataset.

## Motivation

After reading Meta's engineering post on [scaling Instagram Explore](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/), I wanted to understand how large-scale recommendation systems actually work under the hood, not just at a conceptual level, but by building and measuring one myself.

Instagram Explore handles recommendations at a scale where it's impossible to compare every user against every piece of content. Their solution is a two-stage pipeline:

1. **Retrieval (First-stage):** A lightweight model quickly narrows millions of candidates down to a few hundred relevant ones.
2. **Ranking (Second-stage):** A heavier model re-ranks those candidates with higher precision.

This project implements the **first stage** using a Two-Tower architecture on MovieLens 100K, a contained environment to validate whether the approach actually learns meaningful user preferences.

## Design Decisions

### Why implicit feedback instead of ratings?

A natural first instinct is to train directly on star ratings. But rating values introduce subjective noise, a 3-star from one user might mean the same as a 5-star from another. More importantly, for a retrieval system, what matters most is whether a user is interested in a piece of content at all, not how precisely they'd score it.

Reading DoorDash's approach to [balancing technical rigor with business impact](https://careersatdoordash.com/blog/data-scientists-technical-skill-business-impact/) reinforced this thinking: good model decisions should reflect real behavioral signals, not noisy proxies.

So instead of regressing on ratings, we simplify to **implicit feedback**: watched = +1, not watched = -1. For each positive pair, 8 unseen items are sampled as negatives (ratio 1:8) to give the model enough contrast to learn meaningful separation.

## Architecture

At a high level: two separate neural networks learn representations for users and movies independently. After training, recommendations are made by finding movies whose learned vectors are closest to a given user's vector.

**User Tower** takes a `user_id` and learns a vector that captures that user's taste, purely from their watch history, no demographic data needed.

**Item Tower** takes a `movie_id` plus its genre tags, and learns a vector that captures the movie's characteristics. Genre embeddings are mean-pooled and concatenated with the item embedding before passing through the network.

Both towers are trained jointly. Watched pairs are pushed together, unmatched pairs are pushed apart. After training, movie vectors are pre-indexed in ChromaDB and served via a FastAPI endpoint.

The demo uses three users with clearly different tastes, sci-fi, animation, and crime, to make personalization visible at a glance.

## Demo

![Demo Screenshot](assets/demo.png)

## Experiments

Evaluated using Recall@K on a held-out 20% test split per user. Recall@K measures what fraction of a user's held-out movies appear in the top-K recommendations.

Baseline: 100K dataset, no genre features, 20 epochs.

| Dataset | dim | neg | epoch | Genre | Recall@10 | Recall@20 | Recall@50 |
|---------|-----|-----|-------|-------|-----------|-----------|-----------|
| 100K | 128 | 4 | 20 | ✗ | 0.043 | 0.096 | 0.263 |
| 100K | 128 | 8 | 20 | ✗ | 0.070 | 0.151 | 0.336 |
| 100K | 128 | 8 | 20 | ✓ | 0.061 | 0.135 | 0.325 |
| 1M   | 128 | 8 | 20 | ✓ | 0.031 | 0.058 | 0.159 |
| **100K** | **128** | **8** | **50** | **✓** | **0.078** | **0.156** | **0.353** |

**Recall@50 = 0.353** means the model correctly includes about 35% of what a user would actually watch in 50 candidates, a reasonable net for a retrieval stage that feeds into a downstream ranker.

Key findings:
- A 1:8 negative ratio provided sufficient contrast for the model to learn meaningful separation, doubling from 1:4 gave the largest single improvement across all K values.
- Genre embedding improved results, but required sufficient training epochs (20 → 50) to learn genre semantics alongside behavior signals.
- 1M scored lower due to a larger item pool (3,706 vs 1,682), making top-K retrieval inherently harder with the same architecture.

## Reflections & Next Steps

Building this made clear where the real complexity lies in production recommendation systems, it's not the model architecture itself, but the signal quality, scale, and evaluation discipline around it.

If I were to continue:
- **Enrich the pipeline** — add user-side sequential features and a second-stage ranking model to improve both input quality and final precision.
- **Validate at scale** — offline Recall@K has limits. Testing on a larger dataset with A/B evaluation would be the only way to confirm real-world impact.

## References

- [Scaling the Instagram Explore Recommendations System, Meta Engineering (2023)](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/), motivation for the two-stage retrieval + ranking architecture.
- [Data Scientists: Technical Skill Meets Business Impact, DoorDash Careers Blog](https://careersatdoordash.com/blog/data-scientists-technical-skill-business-impact/), grounded the decision to prioritize behavioral signals over raw ratings.
- [MovieLens 100K Dataset, GroupLens](https://grouplens.org/datasets/movielens/100k/)
