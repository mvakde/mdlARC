# Default cached Drive authentication for Colab

This is the default entry point for new Drive-backed Colab experiments. Use
`scripts/colab_oauth/launch.py` after creating a fresh, unused VM. The chooser's
`--backend colab` uses this entry point; `--backend colab-oauth` is an alias.
The original manual authentication remains an explicit fallback. Both modes run
experiments through `scripts/run_on_colab.py`; backups, checkpoint validation,
saved kernel identity, watcher reconnect, result recovery, and verified shutdown
use the existing code. [Vast](../../workflows/vast/README.md) is the other default
remote workflow and uses its own separate Drive credential.

Use OAuth for the real `your-drive-account@example.com` account and its existing Drive.
No service account is needed. A service account is a separate software identity,
and it cannot own files or consume personal Gmail Drive storage quota. It is not
another name for the experiments Gmail account.

## Setup from scratch (once, then about every seven days)

1. Open [Google Cloud Console](https://console.cloud.google.com/) and create a
   separate project, for example `mdlarc-experiment-drive`. Enable **Google Drive
   API** under APIs & Services → Library. Do not change your existing Colab CLI
   project or credentials.
2. In **Google Auth Platform → Audience**, select **External**, leave publishing
   status **Testing**, and add `your-drive-account@example.com` as a test user. Configure
   the required app name/contact fields in Branding. No organization or public
   verification is required to use a Testing app with an added test user.
3. In Data Access, add `https://www.googleapis.com/auth/drive`. This allows the
   dedicated account to use existing experiment assets and checkpoints from the
   old workflow. It is not limited to files created by the new OAuth app.
   Do not add Gmail, Cloud Platform, or Colab scopes. The code rejects any actual
   granted scopes beyond Drive and checks the Drive account before saving or
   transferring credentials.
4. In **Clients → Create client**, choose **Desktop app**. Store its downloaded
   JSON **outside this repository** at
   `~/.config/mdlarc-oauth-clients/mdlarc-oauth-client.json`, moving it out of
   Downloads. Use this location on both laptops. Set the containing directory's
   permissions to `700` and the file's to `600`. This identifies the OAuth app;
   it is not your Google password. Keep it for future browser reauthorization;
   do not copy it into Git or send its contents in chat. The separate Vast
   client is `~/.config/mdlarc-oauth-clients/mdlarc-vast-oauth-client.json`.
5. From the repository root, run:

   ```bash
   uv pip install --python venv/bin/python -r scripts/requirements_colab_drive.txt
   venv/bin/python scripts/colab_oauth/setup.py \
     --client-secrets ~/.config/mdlarc-oauth-clients/mdlarc-oauth-client.json
   ```

   Choose **your-drive-account@example.com** in the browser and grant Drive access. A local
   callback receives the code automatically; no code needs to be copied into
   the terminal. Selecting another account fails before any token is saved.
   This command performs read-only identity/scope checks; it creates no Drive
   files and allocates no Colab VM.

The private refresh token is saved at
`~/.config/mdlarc-colab-oauth/your-drive-account/token.json` with file permissions `600`
and containing-directory permissions `700`. It is a separate credential from
the Colab CLI login and from the historical uploader's token. Source code only
contains public configuration, including the experiment email address.

Testing refresh grants normally expire after seven days. To renew, repeat:

```bash
venv/bin/python scripts/colab_oauth/setup.py \
  --client-secrets ~/.config/mdlarc-oauth-clients/mdlarc-oauth-client.json --reauthorize
```

The previous token is replaced only after the new grant passes account and scope
checks. This same token is used by `venv/bin/mdlarc-records sync`. Moving the
OAuth client JSON from Downloads to its private config location does not require
reauthorization or change the cached token. Renewing the Mac token file does not
change credentials on an already-running VM. Long runs crossing the Testing
expiry may lose Drive access even though
ordinary short-lived access tokens are refreshed automatically. Inspect retained
VMs and backup ledgers using the existing recovery procedure before stopping.

Check a saved token without accessing any VM:

```bash
venv/bin/python scripts/colab_oauth/authorize.py --check
```

## Default launch: cached authentication

Use a **fresh session name and unused VM for each experiment**:

```bash
colab new -s oauth-test-1 --gpu A100 --high-mem
venv/bin/python scripts/colab_oauth/launch.py \
  --config specs/conceptarc_smoke.json --session oauth-test-1 \
  --intent "Cached OAuth smoke test" \
  --changed-from-previous "Testing automatic experiment-account authentication"
```

`--auth-mode cached` is the default. Select the Colab compute login with
`--account 1` (`your-colab-account@example.com`, default) or `--account 2`
(`your-drive-account@example.com`). Create account 2's VM using
`venv/bin/python scripts/colab_accounts.py --account 2 -- new -s SESSION --gpu A100`,
then pass `--account 2` to the launch wrapper. The helper prompts for account
2's own Colab CLI consent when needed and preserves account 1's existing login.
See the [Colab account guide](../README_colab.md) for setup, native chooser,
concurrent runs, and recovery. Both compute accounts use the same verified
Drive-only experiment credential for backups. The wrapper validates the spec,
refreshes and verifies the saved **experiment-account** token, authorizes the
VM, then invokes the existing launcher. Until the Testing grant expires, there
are no per-VM Drive login, consent, or verification-code prompts. All normal
launcher options pass through, including continuation/checkpoint flags and
`--stop-after-success`.

Or keep the two steps explicit for easier testing:

```bash
colab new -s oauth-test-2 --gpu A100 --high-mem
venv/bin/python scripts/colab_oauth/authorize.py --session oauth-test-2
venv/bin/python scripts/run_on_colab.py \
  --config specs/conceptarc_smoke.json --session oauth-test-2 \
  --intent "Separate OAuth authorization smoke test" \
  --changed-from-previous "Prepared VM credentials before the normal launcher"
```

Do not run cached authorization on a VM already manually authenticated by
`colab auth`. It deliberately refuses to replace those credentials. It also
refuses any VM containing mdlARC run files and any session with existing local
run records. Authentication never allocates, stops, or replaces a VM, and an
ambiguous installation never proceeds to training. After an interrupted
authentication, inspect that same session; repeating authorization on an unused
VM is allowed only when the existing credentials are verified as belonging to
this cached-authentication workflow. Successful installation is acknowledged
explicitly.

## Manual authentication fallback

Use this when manual sign-in is explicitly selected, or choose
`--backend colab-manual` in the workflow chooser after authenticating the VM:

```bash
colab new -s manual-test-1 --gpu A100 --high-mem
venv/bin/python scripts/fix_colab_cli.py
colab auth -s manual-test-1
# Choose your-drive-account@example.com and paste the code as before.
venv/bin/python scripts/colab_oauth/launch.py --auth-mode manual \
  --config specs/conceptarc_smoke.json --session manual-test-1 \
  --intent "Manual authentication smoke test" \
  --changed-from-previous "Explicit legacy authentication fallback"
```

Manual mode reads and transfers no cached Drive token. The original
`scripts/run_on_colab.py` command remains available for already authenticated
VMs. Cached-token failures never fall back to browser sign-in automatically;
renew the token through the setup command above.

## Reconnect and laptop sleep

The VM receives the Drive refresh credential and can renew its access tokens
without the Mac. If the Mac shuts down, remote training and Drive backups can
continue while Colab retains the VM and the Testing grant remains valid. Local
watching, result recovery, and automatic shutdown wait for the Mac. This does
not extend Colab's allocation limits or stop a provider from reclaiming a VM.

Use the existing watcher to return to the same run:

```bash
venv/bin/python scripts/run_on_colab.py --watch-session oauth-test-1
```

`colab_oauth/launch.py --watch-session oauth-test-1` delegates to that same
watcher without reading a local Drive token, transferring credentials, or
submitting training. Follow [the existing recovery guide](../README_colab_drive.md)
before stopping a run with incomplete backups.

## Credential isolation and local tests

Credentials are uploaded separately through `colab upload`, never embedded in
`colab exec` code, command arguments, source tarballs, specs, notes, journals, or
result archives. A private temporary file on the Mac is removed after transfer.
The VM verifies the upload checksum, account, and actual OAuth scopes, then
installs credentials at `/content/.adc/adc.json`, outside `/content/mdlARC`, with
permissions `600`. This is the location the existing Drive client already reads.
The private VM staging directory uses permissions `700`; failed staging cleanup
is best-effort if the connection is lost. Installed credentials remain on that
VM until it is released. Provider error bodies and credential contents are not
printed; CLI history contains generic verification code and transfer paths.

The implementation rejects credential paths anywhere in the checkout, including
ignored folders. `.gitignore` also excludes common credential filenames as a
second defense against accidental copies. These controls do not prevent an
intentional `git add -f` or a user manually pasting secrets into another file.
The existing email addresses and Drive IDs in source/history remain public
configuration if you later publish that history.

Run the targeted tests without login, network access, or VM allocation:

```bash
venv/bin/python -m pytest -q \
  tests/test_colab_oauth.py tests/test_colab_launch_assets.py \
  tests/test_colab_drive.py tests/test_colab_connection.py \
  tests/test_colab_watch.py tests/test_run_records.py
```

Preview the same smoke spec without credentials or allocation:

```bash
venv/bin/python scripts/colab_oauth/launch.py \
  --config specs/conceptarc_smoke.json --dry-run
```

For initial setup or token renewal, start with `--check`. To validate a changed
authentication implementation, use one smoke run and a second fresh-VM smoke
run without reauthorizing, then verify both runs' Drive backups and local result
recovery. Authentication uses the existing Colab launcher for execution and
recovery; existing run folders and uploaded source snapshots stay intact.

Google references: [desktop OAuth and offline refresh](https://developers.google.com/identity/protocols/oauth2/native-app),
[Testing token lifetime](https://developers.google.com/identity/protocols/oauth2),
[service-account Drive storage limitation](https://developers.google.com/workspace/drive/api/guides/handle-errors).
