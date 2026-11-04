# Amazon Rekognition Adapter (Contract v1)

`face_identify` and a face gallery on an
[Amazon Rekognition](https://docs.aws.amazon.com/rekognition/latest/dg/collections.html)
collection. It is a drop-in for the attendance InsightFace
(`adapters/insightface`): the same `face_identify` result and the same
gallery routes, so OpenNVR attendance can use either.

> **This sends faces off the site.** Every frame the attendance app looks
> at and every photo enrolled goes to AWS. The adapter declares that
> egress, so KAI-C refuses it under `AI_SOVEREIGNTY=local_only` until an
> operator changes that on purpose. Prefer the local InsightFace adapter.
> Use this only where the site already runs on AWS and the people
> enrolled have agreed to it.

## Tasks

| `task` | Result |
|---|---|
| `face_identify` | Every face in the frame looked up in the collection. Same shape as InsightFace's `face_identify` (bbox, confidence, recognized, person_id, similarity, second_similarity, `frame`, `dropped`) |

The calls behind one look are one `DetectFaces`, then one
`SearchFacesByImage` per face. A 40-face classroom is therefore **41
billed calls per look**. At the attendance default of one look per camera
every 150 s, that is about 1,000 calls per camera per hour.

Similarities come back from AWS on a 0–100 scale and are reported as 0–1.
They are not cosine similarities: a true match is about 0.99. The adapter
never searches below `OPENNVR_REKOGNITION_MIN_SIMILARITY` (default `90`),
whatever threshold the caller sends.

## Gallery routes (called by OpenNVR core for enrolment)

| Route | |
|---|---|
| `POST /faces/register` | multipart `frame`, `person_id`. Indexes one photo; refuses 0 or several faces (422). Always appends. |
| `DELETE /faces/{person_id}` | Deletes every face of the person; 404 if there were none |
| `GET /faces` | `person_id` and sample count per person |

A local index (`OPENNVR_REKOGNITION_INDEX`, a volume) maps each person to
their face ids. It is rebuilt from `ListFaces` at start-up, so it can never
disagree with the collection for long.

## Configuration

| Variable | Default | |
|---|---|---|
| `AWS_REGION` | — | Region of the collection |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | — | Or an instance role. Needs `rekognition:DetectFaces`, `SearchFacesByImage`, `IndexFaces`, `ListFaces`, `DeleteFaces`, `DescribeCollection`, `CreateCollection` on the collection |
| `OPENNVR_REKOGNITION_COLLECTION` | `opennvr-attendance` | Created if missing |
| `OPENNVR_REKOGNITION_MIN_SIMILARITY` | `90` | Floor for a match, 0–100 |
| `OPENNVR_ADAPTER_TOKEN` | — | Bearer the adapter requires (the deployment's `INTERNAL_API_KEY`) |

To use it for attendance, register it with KAI-C and set the attendance
app's **Face adapter** setting to its KAI-C name. Then point core's
`ATTENDANCE_FACE_ADAPTER_URL` at it, so enrolment goes to the same
collection.
