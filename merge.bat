REM Creates an orphaned public branch (no history)
REM git checkout --orhpan public

REM merge to their's accepting all our changes
git merge --no-commit --allow-unrelated-histories --strategy-option theirs master

REM git rm .gitignore


REM removes staged files from the index that match our public .gitignore
git rm -r --cached .
git add .