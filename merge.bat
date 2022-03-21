REM Following command will merge master -> public, first switch to public branch
REM run the following and then review changes before committing.

git merge --no-commit --allow-unrelated-histories --strategy-option theirs master


