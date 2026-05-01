# Two-Tower Retrieval for Recommendation

Retrieval-stage movie recommendation system using a Two-Tower neural 
architecture, trained on MovieLens 100K with implicit feedback and 
genre embeddings.

Through systematic ablation over negative sampling ratio and training 
epochs, Recall@50 improved from 0.263 to 0.353 (+34%). The biggest 
single lift came from doubling negative samples (1:4 → 1:8), suggesting 
that for retrieval, the contrast between what users want and don't want 
matters more than the volume of positive examples — a counterintuitive 
result worth surfacing.

## Motivation

Building [AskUCI](https://github.com/ShengPeiWilliam/askuci) showed that 
pretrained embeddings can handle retrieval well on short, consistent text. 
But they can't adapt to a specific domain or learn from user behavior. I 
wanted to try training embeddings from scratch, and movie recommendation 
felt like the right starting point: well-defined domain, clean datasets, 
and a classic benchmark problem.

Reading Meta's post on [scaling Instagram Explore](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/) 
shaped the approach. Their two-stage pipeline splits the problem into 
retrieval (narrow millions down to hundreds) and ranking (re-rank with 
precision). This project focuses on the first stage using a Two-Tower 
architecture on MovieLens 100K.

## Design Decisions

### Why implicit feedback instead of ratings?

A natural first instinct is to train directly on star ratings. But rating 
values introduce subjective noise — a 3-star from one user might mean the 
same as a 5-star from another. More importantly, for a retrieval system, 
what matters most is whether a user is interested in a piece of content 
at all, not how precisely they'd score it.

So instead of regressing on ratings, the model uses **implicit feedback**: 
watched = +1, not watched = -1. For each positive pair, 8 unseen items 
are sampled as negatives (ratio 1:8) to give the model enough contrast 
to learn meaningful separation.

### Why two separate towers?

User and item representations are learned in independent networks, then 
joined only at the final inner-product step. This separation matters at 
serving time: item vectors can be pre-computed once and stored in a 
vector index (ChromaDB), while only the user vector needs to be computed 
on the fly. This makes retrieval sub-linear in catalog size, which is 
the whole point of a retrieval stage.

### Why mean-pool genre embeddings?

User behavior alone (which movies were watched) gives a sparse signal, 
especially for users with limited history. Adding genre as a content 
feature enriches the item representation: even if two movies have no 
shared viewers, the model can still learn they're similar through 
genre overlap. Mean-pooling handles the variable number of genres per 
movie cleanly, and concatenating the pooled vector with the movie 
embedding lets the model combine behavioral and content signals jointly.

## Demo

![Demo Screenshot](assets/demo.png)

## Experiments

**Headline finding**: Recall@50 improved from 0.263 to 0.353 (+34%) primarily through doubling the negative sampling ratio (1:4 → 1:8) and extending training to 50 epochs. Adding more data (100K → 1M) without adjusting the architecture did not help.

Evaluated using Recall@K on a held-out 20% test split per user. Recall@K 
measures what fraction of a user's held-out movies appear in the top-K 
recommendations.

Baseline: 100K dataset, no genre features, 20 epochs.

| Dataset | dim | neg | epoch | Genre | Recall@10 | Recall@20 | Recall@50 |
|---------|-----|-----|-------|-------|-----------|-----------|-----------|
| 100K | 128 | 4 | 20 | ✗ | 0.043 | 0.096 | 0.263 |
| 100K | 128 | 8 | 20 | ✗ | 0.070 | 0.151 | 0.336 |
| 100K | 128 | 8 | 20 | ✓ | 0.061 | 0.135 | 0.325 |
| 1M   | 128 | 8 | 20 | ✓ | 0.031 | 0.058 | 0.159 |
| **100K** | **128** | **8** | **50** | **✓** | **0.078** | **0.156** | **0.353** |

**Recall@50 = 0.353**: given 50 candidates, the model correctly includes 
35% of what a user would actually watch — a reasonable net for a retrieval 
stage feeding into a downstream ranker.

Key findings:
- Doubling negative samples (1:4 → 1:8) gave the largest single improvement. More contrast in what a user *didn't* watch mattered far more than adding training data.
- Genre embedding showed no advantage at 20 epochs but surpassed the no-genre baseline at 50. The model needs enough training time to learn genre semantics on top of behavioral signals.
- Scaling from 100K to 1M didn't help. A larger item pool makes retrieval harder, and the same architecture without adjustment isn't enough to handle it.

## Reflections & Next Steps

Building this made clear where the real complexity lies in production 
recommendation systems: it's not the model architecture itself, but the 
signal quality, scale, and evaluation discipline around it. The negative 
sampling result is a small example — a hyperparameter choice that 
out-impacts adding 10× more data.

Next steps:
- **Enrich the pipeline**: add user-side sequential features and a second-stage ranking model to improve both input quality and final precision.
- **Validate at scale**: offline Recall@K has limits. Testing on a larger dataset with A/B evaluation would be the only way to confirm real-world impact.
- **Negative sampling strategies**: random negative sampling is a baseline. Hard negative mining (sampling items that are similar but not watched) could improve discrimination at the boundary.

## Tools

**Modeling**: PyTorch, Two-Tower architecture, embedding learning, negative sampling  
**Infrastructure**: ChromaDB (vector index), FastAPI (serving)  
**Language**: Python  
**Libraries**: pandas, numpy, scikit-learn

## References

- [Scaling the Instagram Explore Recommendations System](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/), Meta Engineering (2023) — motivation for the two-stage retrieval + ranking architecture.
- [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/), GroupLens — training dataset, 100,000 ratings from 943 users across 1,682 movies.