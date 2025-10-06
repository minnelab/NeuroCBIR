
# NeuroCBIR — Docker Setup & Usage Guide

This document summarizes the necessary commands to build and run **NeuroCBIR** using Docker and `docker-compose`.

---

## 📁 Project Structure

```
project-root/
├── deploy/
│   ├── configs/
│   ├── data/
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── entrypoint.sh
│   ├── infra/
│   │   └── docker-compose.yml
│   └── neurocbir/
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 🛠 Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (latest stable version)
- [Docker Compose](https://docs.docker.com/compose/install/) (latest stable version)
- Ensure `.dockerignore` exists in the project root to optimize builds

---

## 📌 Building the NeuroCBIR Docker Image

From the project root (where `.dockerignore` is located):

```bash
docker-compose -f deploy/infra/docker-compose.yml build neurocbir
```

This will:
- Set the build context to the project root
- Use the Dockerfile located at `deploy/docker/Dockerfile`
- Build the `neurocbir` container image

---

## 🚀 Running NeuroCBIR and Dependencies

From the project root:

```bash
docker-compose -f deploy/infra/docker-compose.yml up -d
```

This starts:
- `fastsurfer` service
- `neurocbir` service

---

## 📄 Viewing Logs

To view logs for a specific service:

```bash
docker-compose -f deploy/infra/docker-compose.yml logs -f neurocbir
```

Or for `fastsurfer`:

```bash
docker-compose -f deploy/infra/docker-compose.yml logs -f fastsurfer
```

---

## 🛑 Stopping Services

```bash
docker-compose -f deploy/infra/docker-compose.yml down
```

This stops and removes containers while keeping volumes and networks.

---

## 🔄 Rebuilding After Changes

If you change code or configuration files:

```bash
docker-compose -f deploy/infra/docker-compose.yml build --no-cache neurocbir
docker-compose -f deploy/infra/docker-compose.yml up -d
```

---

## 📂 Data & Config Volumes

Volumes defined in `docker-compose.yml` allow you to share files between host and container:

| Host Path        | Container Path      | Purpose                             |
|-------------------|----------------------|--------------------------------------|
| `../data`        | `/data`              | Input/output data files             |
| `../configs`     | `/configs`           | Configuration YAML files            |
| `../app`         | `/app`               | Application source code              |

---

## ⚙ Common Docker Commands

List running containers:
```bash
docker ps
```

Stop a container:
```bash
docker stop neurocbir
```

Remove stopped containers:
```bash
docker container prune
```

List images:
```bash
docker images
```

Remove an image:
```bash
docker rmi neurocbir:latest
```

---

## 📚 References

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

---

**Tip:** Always keep `.dockerignore` up-to-date to avoid unnecessary files in your build context and speed up Docker builds.