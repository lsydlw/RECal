# GitHub publication checklist

- [x] Portable scripts contain no server-specific absolute path.
- [x] Password, SSH host, port, and private-key markers are absent.
- [x] Raw data, caches, checkpoints, results, and logs are ignored.
- [x] Required CSV schema and dataset directory names are documented.
- [x] Every paper experiment has a launcher and output description.
- [x] Validation selection and one-time frozen-test order are documented.
- [ ] Run Python byte-code compilation in the target conda environment.
- [ ] Run `bash -n scripts/*.sh` on Linux/macOS before release.
- [ ] Choose and add the project license before making the repository public.
- [ ] Add dataset source/license URLs appropriate to the exact processed
      versions you are permitted to redistribute.
- [ ] Run the included unit tests after creating the documented environment.
- [ ] Add the final paper title, author list, and BibTeX entry after submission.

Initialize/push commands (the local repository is already initialized on
branch `main`):

```bash
git add .
git commit -m "Release RECal reproduction package"
git remote add origin <your-github-repository-url>
git push -u origin main
```
