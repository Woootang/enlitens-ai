# Enlitens Hosting & Ops Guide

This guide shows how to run the dashboard and ingestion pipeline 24/7 on your
Linux workstation and expose it securely over the internet.

---

## 1. Python services (dashboard + batch ingest)

### Run manually

```bash
cd /home/antons-gs/enlitens-ai
source venv/bin/activate
./scripts/start_dashboard.sh
```

The script launches `gunicorn` on port `5000`.  Stop it with `Ctrl+C`.

To kick off a batch ingest run:

```bash
source venv/bin/activate
python3 scripts/run_ingest_batch.py --model llama --auto-start --auto-stop
```

Useful flags:

- `--limit N` — dry run on the first *N* PDFs.
- `--skip-gemini` — skip Gemini consolidation when testing.

### systemd units

Copy the templated service files to `/etc/systemd/system` and enable them using
your Linux username (e.g. `antons-gs`).

```bash
sudo cp ops/systemd/enlitens-dashboard@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable enlitens-dashboard@antons-gs.service
sudo systemctl start enlitens-dashboard@antons-gs.service
```

The dashboard will now restart automatically on boot and if the process crashes.

To launch a batch ingest on demand:

```bash
sudo cp ops/systemd/enlitens-ingest@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start enlitens-ingest@antons-gs.service
```

> **Tip:** Edit the service file before copying if you want to enable Postgres
> (`ENLITENS_ENABLE_POSTGRES=1`) or Neo4j (`ENLITENS_ENABLE_NEO4J=1`).

Check logs with:

```bash
journalctl -fu enlitens-dashboard@antons-gs.service
```

---

## 2. Cloudflare Tunnel (remote HTTPS access)

1. Install Cloudflare’s lightweight daemon:
   ```bash
   curl -fsSL https://developers.cloudflare.com/cloudflare-one/static/downloads/cloudflared-linux-amd64.deb -o cloudflared.deb
   sudo dpkg -i cloudflared.deb
   ```
2. Authenticate once:
   ```bash
   cloudflared tunnel login
   ```
   Follow the browser prompt to link the machine to your Cloudflare account.
3. Create a named tunnel:
   ```bash
   cloudflared tunnel create enlitens-dashboard
   ```
   This outputs the credentials file path, e.g.
   `/home/antons-gs/.cloudflared/enlitens-dashboard.json`.
4. Write the tunnel configuration (`~/.cloudflared/config.yml`):
   ```yaml
   tunnel: enlitens-dashboard
   credentials-file: /home/antons-gs/.cloudflared/enlitens-dashboard.json

   ingress:
     - hostname: dashboard.yourdomain.com
       service: http://localhost:5000
     - service: http_status:404
   ```
5. (Optional) Protect access with Cloudflare Access — visit the Cloudflare
   dashboard and add an Access policy for the hostname.
6. Launch the tunnel:
   ```bash
   cloudflared tunnel run enlitens-dashboard
   ```
   The dashboard is now available globally via `https://dashboard.yourdomain.com`.

Copy `ops/systemd/enlitens-cloudflared.service` into place and enable it once
your tunnel is configured:

```bash
sudo cp ops/systemd/enlitens-cloudflared.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now enlitens-cloudflared@antons-gs.service
```

The service reads `/etc/enlitens/cloudflared.env` for `CF_TUNNEL_NAME` and
expects the tunnel credentials JSON at `/etc/cloudflared/${CF_TUNNEL_NAME}.json`.
Create that env file before enabling the service:

```bash
sudo mkdir -p /etc/enlitens
echo 'CF_TUNNEL_NAME=enlitens-dashboard' | sudo tee /etc/enlitens/cloudflared.env
```

---

## 3. Optional persistence services

| Service  | Enabled by                            | Notes |
|----------|---------------------------------------|-------|
| Postgres | `ENLITENS_ENABLE_POSTGRES=1`          | Preinstalled locally (PostgreSQL 16) with DB `enlitens`.  Default connection string: `postgresql:///enlitens` (unix socket).  `pgvector` extension is enabled. |
| Neo4j    | `ENLITENS_ENABLE_NEO4J=1`             | Preinstalled locally (Neo4j 5).  Run `sudo -u neo4j neo4j-admin dbms set-initial-password '<your-password>'` and record it in `/etc/enlitens/enlitens.env`.  URI: `bolt://localhost:7687`. |
| Chroma   | Install `chromadb` in the venv        | Optional.  Needed only if you want local vector mirroring.  Some environments may require pinning dependencies manually. |

If a service is disabled or missing packages, the ingestion pipeline logs a
warning and continues without failing the run.

---

## 4. Secrets environment file

Store runtime secrets in `/etc/enlitens/enlitens.env` (outside the git repo):

```bash
sudo mkdir -p /etc/enlitens
sudo nano /etc/enlitens/enlitens.env
```

Example contents:

```
ENLITENS_ENABLE_POSTGRES=1
DATABASE_URL=postgresql:///enlitens
ENLITENS_ENABLE_NEO4J=1
ENLITENS_NEO4J_URI=bolt://localhost:7687
ENLITENS_NEO4J_USER=neo4j
ENLITENS_NEO4J_PASSWORD=your-neo4j-password
```

Reload the services after editing:

```bash
sudo systemctl daemon-reload
sudo systemctl restart enlitens-dashboard@antons-gs.service
```

---

## 5. Backups

Run the snapshot helper any time you want a compressed copy of the ledger,
Chroma store, and logs:

```bash
cd /home/antons-gs/enlitens-ai
./scripts/run_backup.sh
```

Archives are written to the `backups/` directory.  To automate this, add a cron
entry:

```bash
0 3 * * * /home/antons-gs/enlitens-ai/scripts/run_backup.sh >/home/antons-gs/enlitens-ai/backups/last.log 2>&1
```

---

## 6. Daily maintenance checklist

- `tail -f logs/processing.log` during runs to monitor progress.
- Check `enlitens_corpus/failed/` for PDFs that need manual review.
- Back up the ledger and vector store:
  ```bash
  tar -czf backups/enliten-ledger-$(date +%F).tar.gz data/knowledge_base/enliten_knowledge_base.jsonl
  tar -czf backups/enliten-chroma-$(date +%F).tar.gz data/vector_store/chroma
  ```
- Install OS security updates regularly and reboot after kernel upgrades.

---

## 7. Troubleshooting quick hits

- **Dashboard shows zeros** — ensure the service is running (`systemctl status`),
  and that `logs/processing.log` contains recent entries.
- **Cloudflare tunnel down** — restart the service (`systemctl restart enlitens-cloudflared@antons-gs.service`).
- **Postgres failures** — run `psql $DATABASE_URL -c 'SELECT 1'` to validate the DSN.
- **GPU saturation** — use `nvidia-smi` to monitor in real time; adjust `--limit`
  if you want shorter smoke tests.

Keep this file synced with any infrastructure changes so you (and future you)
can recover quickly. 

