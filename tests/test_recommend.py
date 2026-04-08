import json
import requests

API_URL = "http://localhost:8000/recommend"
USERS_PATH = "demo/users.json"
K = 10


def load_users():
    with open(USERS_PATH, "r") as f:
        return json.load(f)


def test_returns_correct_count():
    """Each user should get exactly k recommendations."""
    users = load_users()
    for user in users:
        res = requests.post(API_URL, json={"user_id": user["user_id"], "k": K})
        assert res.status_code == 200, f"user_id={user['user_id']} got {res.status_code}"
        recs = res.json()["recommendations"]
        assert len(recs) == K, f"user_id={user['user_id']} expected {K} recs, got {len(recs)}"


def test_results_have_title():
    """Every recommendation must have a title field."""
    users = load_users()
    for user in users:
        res = requests.post(API_URL, json={"user_id": user["user_id"], "k": K})
        recs = res.json()["recommendations"]
        for rec in recs:
            assert "title" in rec, f"Missing title in rec: {rec}"


def test_personalization():
    """Different users should not get identical recommendations."""
    users = load_users()
    all_titles = []
    for user in users:
        res = requests.post(API_URL, json={"user_id": user["user_id"], "k": K})
        titles = {r["title"] for r in res.json()["recommendations"]}
        all_titles.append(titles)

    # At least one pair of users should have different recommendations
    assert all_titles[0] != all_titles[1] or all_titles[1] != all_titles[2], \
        "All users got identical recommendations — personalization not working"


def test_unseen_user_returns_fallback():
    """Unknown user_id should return fallback results, not 500."""
    res = requests.post(API_URL, json={"user_id": 99999, "k": K})
    assert res.status_code == 200, f"Unseen user got {res.status_code}"
    recs = res.json()["recommendations"]
    assert len(recs) > 0, "Unseen user got empty recommendations"


if __name__ == "__main__":
    tests = [
        test_returns_correct_count,
        test_results_have_title,
        test_personalization,
        test_unseen_user_returns_fallback,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
