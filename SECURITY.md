# Security and privacy

Please do not include camera-trap images, private locations, credentials,
model weights, or other sensitive data in public issues or pull requests.

AmphiLens is local-first. Source images are read from the paths supplied to a
project and are not uploaded by the core package. Optional Docker, CVAT REST,
SSH, Slurm, object-storage, and hosted-worker integrations will need separate
security reviews before use with sensitive wildlife data.

To report a security issue, contact the repository maintainer privately rather
than opening a public issue. Include the affected version, a concise impact
description, and reproducible steps that do not disclose protected data.

The project is experimental until a release explicitly states otherwise.
Users remain responsible for access control, encryption, retention, backups,
and compliance requirements for their field data and infrastructure.
