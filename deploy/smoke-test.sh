#!/usr/bin/env bash
# Prove the deployment works, rather than assume it.
#
#     ./deploy/smoke-test.sh                              # the container, locally
#     ./deploy/smoke-test.sh https://factnet.nayji7.dev   # through the proxy
#
# Run it after every deploy. Exit status is 0 only if everything passed, so it
# can gate a script.
set -uo pipefail

BASE=${1:-http://127.0.0.1:${FACTNET_PORT:-8347}}
PUBLIC=0
case "$BASE" in https://*) PUBLIC=1 ;; esac

pass=0; fail=0
ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=$((fail+1)); }
note() { printf '        %s\n' "$1"; }

echo "== $BASE"
echo

# ---------------------------------------------------------------- the engine
echo "engine"
H=$(curl -sf -m 20 "$BASE/api/health" 2>/dev/null)
if [ -z "$H" ]; then
  bad "/api/health did not answer"
  echo; echo "nothing else can pass while the engine is down."; exit 1
fi

read -r OKF CM GM SM <<<"$(printf '%s' "$H" | python3 -c '
import json,sys
d=json.load(sys.stdin)
print(d.get("ok"), d.get("content_models",0), d.get("graph_models",0), d.get("samples",0))')"

[ "$OKF" = "True" ] && ok "health reports ok" || bad "health reports ok:false"
[ "$CM" -ge 1 ] && ok "content models loaded ($CM)"   || bad "no content model: models/ is empty"
[ "$GM" -ge 1 ] && ok "propagation models loaded ($GM)" || bad "no propagation model"
[ "$SM" -ge 1 ] && ok "stored cascades ($SM)"         || bad "no stored cascade: data/samples.jsonl missing"
[ "$CM" -eq 5 ] || note "5 content models expected, $CM present"

# ---------------------------------------------------------------- the page
echo; echo "page"
code=$(curl -sf -o /tmp/.st_page -w '%{http_code}' -m 20 "$BASE/" 2>/dev/null)
[ "$code" = "200" ] && ok "the page is served" || bad "the page answered $code"
grep -q '<title>' /tmp/.st_page 2>/dev/null && ok "html looks whole" || bad "no <title> in the response"
asset=$(grep -o '/assets/[A-Za-z0-9._-]*\.js' /tmp/.st_page 2>/dev/null | head -1)
if [ -n "$asset" ]; then
  ct=$(curl -sf -o /dev/null -w '%{content_type}' -m 20 "$BASE$asset" 2>/dev/null)
  case "$ct" in *javascript*) ok "the bundle is served ($asset)" ;;
                 *) bad "the bundle answered content-type '$ct'" ;; esac
else
  bad "no bundle referenced in the page"
fi

# ---------------------------------------------------------------- reasoning
echo; echo "readings"
V=$(curl -sf -m 120 -X POST "$BASE/api/verdict" -H 'Content-Type: application/json' \
      -d '{"text":"The Ohio Supreme Court has nullified the $650 million judgement won by multiple counties.","origin":"text"}' 2>/dev/null)
if [ -n "$V" ]; then
  n=$(printf '%s' "$V" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["steps"]))')
  [ "$n" -ge 3 ] && ok "a text reading returns $n stages" || bad "a text reading returned $n stages"
else
  bad "a text reading failed"
fi

C=$(curl -sf -m 180 -X POST "$BASE/api/verdict" -H 'Content-Type: application/json' \
      -d '{"sample_id":0}' 2>/dev/null)
if [ -n "$C" ]; then
  n=$(printf '%s' "$C" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["steps"]))')
  [ "$n" -ge 6 ] && ok "a cascade reading returns $n stages" || bad "a cascade reading returned only $n stages"
else
  bad "a cascade reading failed"
fi

# ---------------------------------------------------------------- streaming
# The one check that a local run cannot fake: if the proxy buffers, every stage
# arrives together at the end and the interface sits blank until then. First
# byte must land well before the last.
echo; echo "streaming"
t0=$(date +%s%N)
first=""
curl -sN -m 180 -X POST "$BASE/api/verdict/stream" -H 'Content-Type: application/json' \
     -d '{"sample_id":0}' 2>/dev/null | while IFS= read -r line; do
  [ -z "$line" ] && continue
  date +%s%N
done > /tmp/.st_times
if [ -s /tmp/.st_times ]; then
  tf=$(head -1 /tmp/.st_times); tl=$(tail -1 /tmp/.st_times)
  ms_first=$(( (tf - t0) / 1000000 )); ms_last=$(( (tl - t0) / 1000000 ))
  lines=$(wc -l < /tmp/.st_times)
  ok "the stream delivered $lines chunks"
  note "first at ${ms_first}ms, last at ${ms_last}ms"
  if [ "$ms_last" -gt 1500 ] && [ "$ms_first" -gt $(( ms_last * 7 / 10 )) ]; then
    bad "stages arrive all at once: something is buffering (nginx proxy_buffering)"
  else
    ok "stages arrive as they are computed"
  fi
else
  bad "the stream returned nothing"
fi

# ---------------------------------------------------------------- exposure
if [ "$PUBLIC" -eq 1 ]; then
  echo; echo "exposure"
  host=${BASE#https://}; host=${host%%/*}
  red=$(curl -s -o /dev/null -w '%{http_code}' -m 15 "http://$host/" 2>/dev/null)
  case "$red" in 30[128]) ok "http redirects to https ($red)" ;;
                 *) bad "http answered $red instead of redirecting" ;; esac

  hdr=$(curl -sI -m 15 "$BASE/" 2>/dev/null | tr 'A-Z' 'a-z')
  for h in strict-transport-security content-security-policy x-content-type-options x-frame-options; do
    printf '%s' "$hdr" | grep -q "^$h:" && ok "$h" || bad "$h missing"
  done
  printf '%s' "$hdr" | grep -q '^server: nginx/' && note "the server header still names a version"

  if command -v ss >/dev/null 2>&1; then
    if ss -ltn 2>/dev/null | grep -q "0.0.0.0:${FACTNET_PORT:-8347}\|\[::\]:${FACTNET_PORT:-8347}"; then
      bad "the container port is published on every interface, not just loopback"
    else
      ok "the container port is not exposed outside the host"
    fi
  fi
fi

rm -f /tmp/.st_page /tmp/.st_times
echo
echo "== $pass passed, $fail failed"
[ "$fail" -eq 0 ]
