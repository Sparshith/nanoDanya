#!/bin/sh
# Commits anything new under knowledge/ and nothing else. Run by cron every 3 hours.
cd "$(dirname "$0")" || exit 1
/usr/bin/git add knowledge
/usr/bin/git diff --cached --quiet -- knowledge && exit 0
/usr/bin/git commit -q -m "sweep knowledge notes" -- knowledge
