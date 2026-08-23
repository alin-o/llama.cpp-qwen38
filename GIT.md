# Git standard workflows

## Sync fork master with upstream (run whenever you want):

git fetch upstream
git checkout master
git merge --ff-only upstream/master
git push origin master

## Rebase your work onto the fresh master:

git checkout turboquant
git rebase master
git push --force-with-lease origin turboquant
