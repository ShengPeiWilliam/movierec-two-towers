# movierec-two-towers

A retrieval-stage movie recommendation system built with a Two-Tower model, trained on the MovieLens 100K dataset.

---

## Motivation

> Inspired by [Scaling the Instagram Explore Recommendations System](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/).

Large-scale recommendation systems (e.g., Instagram Explore) typically operate in two stages:

1. **Retrieval (First-stage):** Quickly retrieve hundreds of relevant candidates from a pool of millions using a lightweight model.
2. **Ranking (Second-stage):** Precisely re-rank the retrieved candidates using a heavier model.

This project focuses on the **first stage — retrieval** — and validates the Two-Tower architecture on a single vertical task: movie recommendation.

Unlike social platforms where recommendations must balance engagement across heterogeneous content, a movie recommender operates in a well-defined domain, making it a clean environment to study retrieval quality and embedding behavior.

---

## Design Decisions

> Guided by the principles in [Data Scientists: Technical Skill Meets Business Impact](https://careersatdoordash.com/blog/data-scientists-technical-skill-business-impact/) — Impact, Expertise, Platformization, Accountability, Reliability, and Efficiency.

### Label Design: Reducing Rating Noise

Raw rating values introduce subjective noise — different users apply different standards (a 3-star from one user may equal a 5-star from another). Instead of regressing on rating scores, we reduce the signal to **implicit feedback**:

- **Watched → label = +1**
- **Not watched (negative sample) → label = -1**

This focuses the model on behavioral signals rather than subjective preferences, and pairs naturally with **CosineEmbeddingLoss**, which learns to push user and item embeddings closer for positive pairs and further apart for negative pairs.

### Negative Sampling

For each positive (user, item) pair, we randomly sample **4 unseen items** as negatives (ratio 1:4). This gives the model sufficient contrast to learn meaningful separation in the embedding space without overwhelming the positive signal.

---

## Architecture

```
User ID ──► User Tower (Embedding → Linear → ReLU → Linear → L2 Norm) ──► user_emb ─┐
                                                                                        ├──► CosineEmbeddingLoss
Item ID ──► Item Tower (Embedding → Linear → ReLU → Linear → L2 Norm) ──► item_emb ─┘
```

After training:
- All item embeddings are pre-computed and stored in **ChromaDB** (cosine space).
- At inference, a user embedding is computed on-the-fly and used to query the nearest item vectors.

---

## Pipeline

Each module has a single responsibility, making the system easy to swap out components:

- **Data** — Raw ratings are preprocessed into implicit feedback pairs. Movie metadata (title, genre) is stored separately and only used for display, not training.
- **Model** — The two towers are trained jointly. After training, the item tower is frozen and its outputs are indexed into ChromaDB for fast retrieval.
- **Retrieval** — At inference, only the user tower runs. The resulting vector queries ChromaDB to find the nearest item embeddings.
- **API + UI** — A FastAPI endpoint wraps the retriever. The frontend demonstrates personalization across three users with distinct taste profiles.

---

## Demo

![Demo Screenshot](assets/demo.png)

---

## Generalizability

The architecture is not specific to movies. Any domain with user-item interaction logs (music, books, e-commerce) can adopt the same pipeline by replacing the dataset. The embedding dimension, negative sampling ratio, and loss function are all configurable via `config.py`.

---

## References

- [Scaling the Instagram Explore Recommendations System — Meta Engineering (2023)](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/)
  — Architecture motivation: two-stage retrieval + ranking, and the role of Two-Tower models in first-stage candidate retrieval.

- [Data Scientists: Technical Skill Meets Business Impact — DoorDash Careers Blog](https://careersatdoordash.com/blog/data-scientists-technical-skill-business-impact/)
  — Design principles: Impact, Expertise, Platformization, Accountability, Reliability, Efficiency. These guided decisions such as using implicit feedback over raw ratings, and keeping the architecture domain-agnostic.

- [MovieLens 100K Dataset — GroupLens](https://grouplens.org/datasets/movielens/100k/)
