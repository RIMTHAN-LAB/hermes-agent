# Native session configuration and readback

ACP initialize advertises `mcpCapabilities.http`, `mcpCapabilities.sse`, and
`_meta.hermes.configurationReadback: 1`. HTTP and SSE inputs retain their explicit
transport when registered with the existing native MCP client. Credentials stay
in ACP's protected local configuration, not in readback.

The authenticated ACP connection supports `_hermes/session/configure` with
`{sessionId,generation,instructions,nativeInstructions?}`. Generation is a
nonnegative integer. Each instruction contribution is bounded UTF-8 text up to
1MiB. Bind once before the first prompt in that provider instance. A newly loaded
instance may adopt a newer generation while preserving prior conversation
history. Decreasing generations, active prompts, and changing an already-bound
instance are refused. Identical retries do not append duplicate instructions.

`_hermes/session/configuration` accepts only `{sessionId}`. Configure returns the
same exact session readback. It acknowledges the actual effective instruction
contributions separately by SHA256, the actual selected native skill names,
source origin and leaf content digests, and native MCP connection state from
successful initialize/tools-list in the existing client. Tool names alone are not
connection evidence. A missing capability or lost instruction contribution is
explicit unavailable evidence. This contract makes no application/authorization
claim for any downstream product.

`nativeConversation` reports a message count and SHA256 of canonical JSON for
actual loaded message `{role,content}` fields only: sorted object keys, compact
separators, UTF-8 with non-ASCII characters preserved. Metadata is omitted. It
refuses more than 10,000 messages, 10MiB encoded data, 32 nested levels or 100,000 values.
This is continuity evidence and returns no transcript. Explicitly configured
empty sessions persist through the existing SessionDB path; production never
adds synthetic history. Saved provider aliases and endpoints survive restoration,
and rebuilding an agent for a model switch retains the exact contributions.

`skills.plugin_dirs: string[]` adds managed package directories, relative to
HERMES_HOME or absolute. Trusted project skills take priority, followed by native
user skills and `external_dirs`, followed by these plugin roots. User learned
`home/skills` remains independent. Plugin roots are read-only package content in
the skill modification path.

A selected directory skill hashes every regular leaf file, including SKILL.md,
references, scripts and assets, with no line-ending normalization or exclusions:
SHA256(`hermes-skill-content-v1` + NUL + sorted file records). Each record is uint32
big-endian UTF-8 relative POSIX path length, path bytes, uint64 big-endian content
length, then raw content. Directories and modes are omitted. Links, depth over 24,
more than 1,000 files/10,000 entries, or total content over 10MiB are refused. Metadata
reads preserve the existing first 4,000 characters and full skill reads are bounded
at 10MiB.

Tests use actual resolver/import paths and explicitly identified local fixtures.
The BB owning qualification script exercises real Hermes ACP/native MCP processes,
shared workdir profile A→B→A, rejected HTTP authentication, and restoration of a
locally seeded historical fixture without a model turn. That remains local
provider preflight evidence; it does not establish downstream production delivery.
