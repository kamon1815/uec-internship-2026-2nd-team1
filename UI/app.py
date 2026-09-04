import os

def write1(file1, str1):
    with open(file1, 'w', encoding='utf-8') as f1:
        f1.write(str1)
    return 0


str1 = '''<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title1}</title>
</head>
<body style="padding: 150px;">
  {body1}
  <video id="myVideo" controls>
    <source src="mask.mp4" type="video/mp4" />
  </video>
</body>

<style>
#myVideo {{
  width: 640px;
  aspect-ratio: 5 / 4;
}}
</style>
</html>'''.format(title1="hello html", body1="hello html!")

print(str1)

path1 = os.path.dirname(os.path.abspath(__file__)) + "/"
file1 = path1 + "html1.html"
write1(file1, str1)
