# Running the dashboard on a server

The container answers both the API and the page, so nginx in front is a plain
reverse proxy with nothing to route. It listens on the loopback interface only.

## What `git pull` does not bring

`models/` and `data/` are not versioned, deliberately: together they are 1.9 GB
and they change far less often than the code. They are pushed once from the
workstation and mounted read-only into the container, so a later code fix is a
rebuild and nothing else.

## First install

On the server, as a user in the `docker` group:

```bash
sudo mkdir -p /opt/factnet && sudo chown "$USER" /opt/factnet
git clone git@github.com:NayJi7/FactNet.git /opt/factnet
cd /opt/factnet
cp env.example .env && chmod 600 .env
```

From the workstation, send the weights and the collected sample:

```bash
./deploy/push-payload.sh user@vps           # one transformer, ~490 MB
./deploy/push-payload.sh user@vps --all     # all five, ~1.9 GB
```

The dashboard reports missing models rather than failing, so the smaller set is
a real option: the selector simply offers one content model instead of five.

Back on the server:

```bash
cd /opt/factnet
docker compose up -d --build
docker compose ps          # wait for "healthy", about 90 seconds on a cold start
curl -s localhost:8347/api/health
```

## Updating

```bash
cd /opt/factnet && git pull && docker compose up -d --build
```

Nothing re-transfers: the weights live outside the image.

## nginx

```bash
sudo cp deploy/nginx/factnet.conf /etc/nginx/sites-available/factnet
sudo sed -i 's/factnet.example.org/YOUR.DOMAIN/g' /etc/nginx/sites-available/factnet
sudo ln -s /etc/nginx/sites-available/factnet /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d YOUR.DOMAIN
```

The configuration turns proxy buffering off on `/api/verdict/stream`. That is
not optional: the reading is streamed stage by stage, and buffering would hold
every stage until the last one, which is the behaviour the streaming exists to
avoid. Read timeouts are 120 s because a live Bluesky fetch takes about 15 s and
a large cascade about 8 s.

## Firewall

Docker writes its own iptables rules and can publish a port straight past ufw.
That cannot happen here because the port is bound to `127.0.0.1`, but the host
should still only accept what it needs:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
sudo ss -ltnp | grep 8347      # expect 127.0.0.1:8347 and nothing else
```

## Logs

Rotated by the daemon at 20 MB across 5 files, so an unattended box cannot fill
its disk with access lines.

```bash
docker compose logs -f dashboard        # live
docker compose logs --since 1h dashboard
```

nginx keeps its own at `/var/log/nginx/factnet.{access,error}.log`.

## What was measured, not guessed

| | |
|---|---|
| resident memory, one worker, all five content models | 2.1 GB |
| the same with only the primary model loaded | ~0.9 GB |
| a text verdict, warm | ~1 s |
| a 581-account cascade | ~8 s |
| the first reading of a session | ~7 s longer, loading weights |

The container is capped at 4 GB and 3 CPUs. Raise `FACTNET_WORKERS` only with
the memory to match: each worker holds its own copy of the weights.

## Hardening in place

Verified on a running container, not merely configured: unprivileged user
(uid 10001), read-only root filesystem with `/tmp` and the cache on capped
tmpfs, all Linux capabilities dropped, `no-new-privileges`, `models/` and
`data/` mounted read-only, published on the loopback interface only.

CORS is closed by default. The page and the API share one origin, so the
browser never makes a cross-origin call and nothing has to be allowed.

## Credentials

Only the "Bluesky link" input needs any: it calls an authenticated endpoint.
Use an application password from the Bluesky settings, never the account
password, and keep `.env` at mode 600. The other three input modes work
without it.
