# adapter/database/ -- Face Storage

Storage backend for face embeddings used by face recognition and watchlist tasks.

## Files

```
database/
└── face_db.py    # FaceDatabase class
```

## face_db.py

### Current: In-Memory Storage

The active `FaceDatabase` class stores everything in a Python dictionary. Fast for testing but **data is lost on restart**.

**Operations:**
- `register_face(person_id, name, embedding, category, metadata)` -- store a 512-dim face embedding
- `get_face(person_id)` -- retrieve face metadata (without embedding)
- `list_faces(category=None)` -- list all faces, optionally filtered
- `delete_face(person_id)` -- remove a face
- `search_similar(query_embedding, threshold, top_k, category)` -- cosine similarity search
- `get_best_match(query_embedding, threshold)` -- return top-1 match

**Cosine similarity** is used for face matching. Default threshold is 0.5 (higher = stricter match).

### Planned: MySQL Storage

The file contains a commented-out MySQL implementation using `mysql.connector`. When enabled, it will:
- Store embeddings as BLOBs in a `faces` table
- Support persistent storage across restarts
- Use in-memory caching for fast search
- Require `FACE_DB_USER` and `FACE_DB_PASSWORD` environment variables

### Categories

Faces are tagged with a `category` string. Common categories:
- `employee` -- known staff
- `visitor` -- authorized visitors
- `watchlist` -- persons of interest (triggers alerts)
- `vip` -- high-priority persons
- `unknown` -- default if not specified

The `watchlist_check` task filters searches to only match `category="watchlist"`.
