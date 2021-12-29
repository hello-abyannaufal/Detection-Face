from algorithm import FaceDetection

print('====================================[WELCOME TO FACE DETECTION]====================================\n'
      '1. Image - Face Detection\n'
      '2. Video - Face Detection')
select = int(input('Select mode: '))

FaceDetection(select)