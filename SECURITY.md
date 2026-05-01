# Security Policy

## Reporting Security Vulnerabilities

We take security seriously and appreciate your efforts to responsibly disclose any vulnerabilities.

### How to Report

**Please do NOT open public issues for security vulnerabilities.**

Instead, report them via one of the following methods:

1. **GitHub Private Vulnerability Reporting** (preferred):
   - Go to the [Security Advisories](https://github.com/H0NEYP0T-466/cuda-mpi-matmul-engine/security/advisories) page
   - Click "Report a vulnerability"
   - Fill in the details privately

2. **GitHub Issues** (for non-sensitive concerns):
   - Open an issue with the `security` label for general security questions

### What to Include

- **Description** — Clear description of the vulnerability
- **Impact** — What could an attacker do?
- **Steps to Reproduce** — How to demonstrate the issue
- **Affected Components** — Which files/functions are involved
- **Suggested Fix** — If you have one (optional)

### Response Timeline

| Stage | Timeframe |
|-------|-----------|
| Initial acknowledgment | Within 48 hours |
| Investigation & assessment | Within 7 days |
| Fix or mitigation | Within 30 days (critical: 14 days) |
| Public disclosure | Coordinated with reporter |

## Security Considerations for This Project

### Build & Compilation
- This project compiles and executes native code (C, CUDA, MPI)
- Always review source code before building, especially when building from untrusted forks
- The Makefile invokes `gcc`, `nvcc`, and `mpicc` — ensure these are from trusted sources

### Docker
- The Docker MPI cluster runs SSH with root access and disabled host key checking
- **This is intentional for local development only** — do not expose these containers to public networks
- The Docker containers use a private bridge network (`172.28.0.0/16`)

### AWS Deployment
- **Never commit `.pem` key files** — they are listed in `.gitignore`
- The `aws/` directory contains setup scripts that reference cloud credentials
- Always restrict security group rules to your IP when running cloud experiments

### MPI Communication
- MPI communication between Docker containers uses SSH without password authentication
- This is acceptable for local containerized environments but **must not be used in production**

## Known Limitations

- This is a research/educational project, not a production system
- No authentication or encryption for MPI inter-node communication
- No input sanitization beyond CLI argument parsing (trusted-user model)

## Dependency Security

- Python dependencies (`matplotlib`, `numpy`) are pulled from PyPI — use `pip audit` to check for known vulnerabilities
- System dependencies (GCC, CUDA, OpenMPI) should be installed from official repositories
