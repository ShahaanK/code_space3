# Apophis Jupyter Terminal Recovery

Operational note for the Jupyter Lab server on Apophis (port 8891). Written 2026-07-23
after the terminals stopped accepting input. Keep this handy; the problem recurs on
long-lived servers.

## Symptoms

- You cannot type in one or more Jupyter Lab terminals. Keystrokes do nothing, no echo.
- 4 to 5 terminal tabs are "stuck" and will not close from the UI.
- Notebooks and kernels may still work fine; only the terminals are affected.

## What actually happened (2026-07-23)

The Jupyter Lab server (PID 3912104) had been running continuously since March 1,
about 143 days. Jupyter's terminal subsystem (a library called terminado) tracks each
open terminal and the shell process behind it. Over a long uptime, with network blips,
laptop sleeps, and reconnects, that bookkeeping drifted out of sync with reality.

Concretely, at the time of the incident:

- The server's terminal manager was tracking 6 terminals.
- Only 2 of them still had a live `bash` shell behind them.
- The other 4 were "zombies": the browser tab still existed and kept trying to
  reconnect, but there was no shell process on the other end. One of them had last had
  real activity on May 28, almost two months earlier.

Because a zombie terminal has no shell behind it:

- Typing goes nowhere, so it looks frozen.
- It will not close cleanly, because terminado cannot reap a session whose underlying
  pseudo-terminal (PTY) is already gone.

The rest of the machine was healthy the whole time: plenty of memory, low CPU load, no
stuck processes, PTY usage far below the system limit. The problem was isolated to this
one server's terminal manager.

## Why the quick fixes did not work

1. Browser refresh alone does not help once the backend session is truly dead. It can
   re-establish a websocket, but there is no shell to connect to.

2. Deleting the terminals through the Jupyter REST API did not clear the zombies. The
   API returned "success" (HTTP 204) for every delete, and the 2 live terminals did go
   away cleanly, but the 4 zombies stayed in the list with their old timestamps
   unchanged. The terminal manager was wedged and could not physically reap them.
   Separately, the still-open browser tabs kept recreating some terminal entries on
   reconnect, which added to the confusion.

The conclusion: once terminado is wedged like this, the API cannot recover it. The only
reliable fix is to restart the Jupyter Lab server.

## The fix: restart the Jupyter Lab server

Restarting clears all terminal state. Important facts before you do it:

- It does NOT delete any files. Killing a running program does not touch the filesystem.
  Every notebook, script, config, and data file stays exactly where it is on disk.
- It DOES reset running kernels, meaning variables and dataframes held in memory are
  lost. The notebook files themselves keep all their code and their last saved outputs.
  You just re-run cells to rebuild in-memory state.
- Before restarting, click File then Save All in the browser so any unsaved edits are
  written to disk.
- On a shared machine like Apophis, only ever target your own server. Confirm the PID is
  owned by szkhan before killing it. Never signal another user's process.

### Step by step

Find your server and confirm you own it:

```bash
source ~/myenv/bin/activate
jupyter server list                 # shows the URL, port, and token
ps -o pid,user,cmd -p <PID>         # confirm USER is szkhan before doing anything
```

Stop your server (replace <PID> with the number for your server only):

```bash
kill <PID>                          # graceful stop
# if it hangs (a wedged terminal manager can block a clean exit):
kill -9 <PID>                       # force stop, still only your one PID
```

Verify it is gone and the port is free:

```bash
kill -0 <PID> 2>/dev/null && echo "still alive" || echo "gone"
ss -ltnp | grep ':8891 ' && echo "port still bound" || echo "port free"
```

Relaunch from the project directory with the same port and token so your bookmarked URL
keeps working. The token below is the local, non-secret token for this Apophis server:

```bash
cd /home/szkhan/code_space3
source ~/myenv/bin/activate
setsid nohup jupyter-lab --port 8891 \
  --IdentityProvider.token=80b984fd04b58bd65c905a0c787c853d8759e517f4e7491b \
  > nohup.out 2>&1 < /dev/null &
```

`setsid` fully detaches the process so it survives your SSH logout (its parent becomes
PID 1). Confirm it came up:

```bash
jupyter server list
curl -s "http://localhost:8891/api/terminals?token=80b984fd04b58bd65c905a0c787c853d8759e517f4e7491b"
# an empty list [] means all terminals are cleared
```

Finally, in the browser, hard-refresh Jupyter Lab (Ctrl+Shift+R) so it drops the stale
tabs and reconnects to the fresh server. Close any dead terminal tabs that linger; they
have no backend now. Open a new terminal and it will accept input normally.

## Housekeeping

- The `nohup.out` log grows without bound over a long uptime. On 2026-07-23 it was about
  78 MB after 143 days. When you restart, rename the old one with a dated name rather than
  letting it grow forever, for example `nohup.out.pre-restart-2026-07-23`.

## How to avoid it

- If a terminal ever stops accepting input, do not fight it through the UI. Restart the
  server using the steps above.
- Consider restarting your Jupyter Lab server occasionally (for example monthly) rather
  than letting it run for many months, which is when this drift builds up.
