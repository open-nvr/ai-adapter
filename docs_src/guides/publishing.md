# Publishing

## Build the image

```bash
docker build -t ghcr.io/you/my-model:1.0.0 .
docker push ghcr.io/you/my-model:1.0.0
```

The scaffolded Dockerfile healthchecks the contract's own `/health`, so
the orchestrator and KAI-C agree about when the adapter is ready.

## Generate the listing

```bash
opennvr-adapter listing . --image ghcr.io/you/my-model:1.0.0 \
    --source https://github.com/you/my-model
```

It reads your adapter's own `/capabilities`, so the entry cannot claim a
task the adapter does not advertise, a permission it does not request,
or a fingerprint it does not compute. What it cannot know is left as
`TODO`: the summary, the model card, the contact.

It also warns about the three things that make a listing useless:

- no advertised task — nothing will ever route work to it;
- a null fingerprint — no drift detection;
- undeclared-in-the-summary egress — the operator is the one being asked
  to allow it.

## Open the pull request

Add the entry to
[`server/config/adapters_index.yml`](https://github.com/open-nvr/open-nvr/blob/main/server/config/adapters_index.yml)
in open-nvr. CI runs `scripts/validate_adapters_index.py`, which checks
the shape and that every advertised task is a real convention.

The full deal, and what a review looks at, is
[CONTRIBUTING_ADAPTERS.md](https://github.com/open-nvr/open-nvr/blob/main/docs/CONTRIBUTING_ADAPTERS.md).

## You do not have to be listed

An adapter is a container that answers HTTP. An operator can run yours
and point KAI-C at it without anyone's permission — the catalog is
discovery, not a gate.

## The licence

`opennvr-adapter-sdk` is Apache-2.0 and talks to the platform over HTTP;
nothing links. Ship your adapter under any licence — open, proprietary,
or classified — and charge whatever you like. OpenNVR takes no fee.
