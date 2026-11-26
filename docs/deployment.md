# Deployment constraints

The first release is a public read-only demo. Use a single Dockerized FastAPI service with CPU-only execution and PostgreSQL with pgvector. Host the immutable source corpus outside the web service's writable filesystem. Keep database and provider credentials in the hosting platform's secret configuration.

Render is the candidate for the application. Its free service publishes an HTTPS URL, uses ephemeral storage, and can spin down during idle periods. Measure peak memory with both CPU retrieval models loaded against the 512 MB limit, and measure cold-start latency before the release decision. [Render web services](https://render.com/docs/web-services) · [Free instance limits](https://render.com/docs/free) · [Compute pricing](https://render.com/pricing)

Neon is the managed database candidate because it supports pgvector. Use TLS and a pooled connection. Keep full PDF files out of PostgreSQL. The local integration target is PostgreSQL with pgvector in Docker Compose. [Neon pgvector](https://neon.com/docs/extensions/pgvector)

Select model providers after measuring their quality, latency, and estimated per-query cost on the development cases. Bound input length, output tokens, retrieval size, and concurrent requests. The public release requires an enforced requests or spend allowance with a defined user-facing response when it is exhausted. Check that outbound provider inputs contain masked sensitive identifiers and that telemetry excludes raw queries.

Release verification includes a successful container build, a health check independent of the model provider, a public smoke test of all three answering paths, and a redeploy or rollback exercise against a known revision. Hosting and database provisioning are pending.
