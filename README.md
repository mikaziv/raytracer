
# exaple commands to run the project:
specific height and width:
```bash
python ray_tracer.py scenes\Spheres.txt scenes\Spheres.png --width 500 --height 500
```

with default height and width (which is 500 pixels for each):
```bash
python ray_tracer.py scenes\Spheres.txt scenes\Spheres.png
```

example:
```bash
python ray_tracer.py scenes\one_sphere.txt output\one_sphere.png
```

testing for pool file but not overriding pool image given in skeleton:
```bash
python ray_tracer.py scenes\pool.txt output\pool_replication_attempt.png
```
ease of use commands for me:
```bash
# connect with https if ssh key not in computer
git clone REPO  

git status

git add file.txt file.png

git commit -m "msg"

git push -u origin main

git pull
```

run testers in CMD:
```bash
New-Item -ItemType Directory -Force output; Get-ChildItem scenes -Filter *.txt | ForEach-Object { python ray_tracer.py $_.FullName "output/$($_.BaseName)_result.png" --height 50 --width 50 }
```

render all testers in parallel in powershell:
```bash
New-Item -ItemType Directory -Force output; Get-ChildItem scenes -Filter *.txt | ForEach-Object { python ray_tracer.py $_.FullName "output/$($_.BaseName)_result.png" --height 500 --width 500 }
```





