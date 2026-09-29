# Security and privacy

Please do not include camera-trap images, private locations, credentials,
model weights, or other sensitive data in public issues or pull requests.

AmphiLens is local-first. Source images are read from the paths supplied to a
project and are not uploaded by the core package. The optional Modal cloud
training workflow uploads the selected labelled dataset and any selected base
checkpoint only after explicit user consent. Modal credentials are stored in a
user-scoped file with owner-only permissions and never enter project manifests,
logs, or command-line arguments. Review Modal's retention, region, privacy, and
billing terms before uploading sensitive wildlife data. Delete remote job data
from the Modal volume after results have been verified; provider deletion may
not immediately remove billable storage.

The cloud cost estimate is a planning range based on an unverified runtime
heuristic. Its budget-derived server-side deadline limits one run but does not
guarantee the provider's final bill, including startup time or interruptions.
The GPU service requires a valid payment method. Institutional SSH, Slurm,
object-storage, and hosted multi-user integrations remain future work.

To report a security issue, contact the repository maintainer privately rather
than opening a public issue. Include the affected version, a concise impact
description, and reproducible steps that do not disclose protected data.

The project is experimental until a release explicitly states otherwise.
Users remain responsible for access control, encryption, retention, backups,
and compliance requirements for their field data and infrastructure.
